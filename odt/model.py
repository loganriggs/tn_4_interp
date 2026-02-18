"""
χ-net and residual bilinear architectures.

The χ-net uses bilinear layers with a cloning operator (no residual).
The residual variant adds skip connections: h_{i+1} = h_i + bilinear(h_i).
"""

import torch
import torch.nn as nn
import math


class RMSBatchNorm(nn.Module):
    """
    Normalize by the batch-averaged L2 norm.
    After training, the running average is folded into adjacent weights.
    From the paper: "a variant of BatchNorm that only divides by the L2 norm
    (akin to RMSNorm). After training, this single average is contracted into
    its neighbouring matrix."
    """
    def __init__(self, dim, momentum=0.1, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_rms', torch.ones(1))

    def forward(self, x):
        if self.training:
            # RMS over features, averaged over batch
            rms = x.norm(dim=-1).mean()
            with torch.no_grad():
                self.running_rms = (1 - self.momentum) * self.running_rms + self.momentum * rms
            return x / (rms + self.eps)
        else:
            return x / (self.running_rms + self.eps)


class BilinearCore(nn.Module):
    """
    Single bilinear core: f(h) = D(Ah ⊙ Bh)

    Parametrized as Khatri-Rao product of A and B, producing a
    third-order tensor with Kronecker delta structure.

    Args:
        dim_in: input dimension (H_i)
        dim_out: output dimension (H_{i+1})
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.A = nn.Linear(dim_in, dim_out, bias=False)
        self.B = nn.Linear(dim_in, dim_out, bias=False)

        # Initialize so that outputs have unit variance
        nn.init.normal_(self.A.weight, std=1.0 / math.sqrt(dim_in))
        nn.init.normal_(self.B.weight, std=1.0 / math.sqrt(dim_in))

    def forward(self, x):
        return self.A(x) * self.B(x)

    def get_interaction_tensor(self):
        """
        Return the full 3rd-order interaction tensor.
        T_{o,i,j} = A_{oi} * B_{oj}  (Khatri-Rao structure)

        This is the tensor such that f(x)_o = sum_{i,j} T_{oij} x_i x_j
        """
        A = self.A.weight  # (dim_out, dim_in)
        B = self.B.weight  # (dim_out, dim_in)
        # T_{oij} = A_{oi} * B_{oj}
        return torch.einsum('oi,oj->oij', A, B)

    def symmetrize_(self):
        """
        Symmetrize the core: T_{oij} <- (T_{oij} + T_{oji}) / 2
        For Khatri-Rao parametrization, this means:
        A', B' such that A'_{oi}B'_{oj} = (A_{oi}B_{oj} + A_{oj}B_{oi})/2

        The simplest approach: compute full tensor, symmetrize, then
        re-fit A, B via SVD of the matricized symmetric tensor.
        """
        T = self.get_interaction_tensor()
        T_sym = 0.5 * (T + T.transpose(1, 2))
        dim_out, dim_in, _ = T_sym.shape
        # Matricize as (dim_out * dim_in, dim_in) and take SVD? No —
        # for Khatri-Rao we need T_{o,i,j} = sum_k U_{o,k} V_{ki} W_{kj}
        # with k = dim_out (diagonal in output).
        # Simplest: just average A and B.
        # A'_{oi}B'_{oj} = ((A+B)/2)_{oi} * ((A+B)/2)_{oj}  — no, that's wrong.
        # Actually for Khatri-Rao, symmetrization means:
        # new_A = (A + B) / 2, new_B = (A + B) / 2  — but this loses rank.
        # Better: keep both and note that symmetry is enforced at analysis time.
        # The paper says "we assume this symmetry is enforced" — for analysis
        # we just symmetrize the interaction tensor directly.
        pass


class ChiNet(nn.Module):
    """
    χ-net: Compositionally and Hierarchically Interpretable network.

    Architecture:
        x ← (1, x)           # append bias
        h = embed(x)          # linear embedding I → H1
        for each layer:
            h = norm(h)
            h = bilinear(h)   # clone + bilinear core H_i → H_{i+1}
        y = unembed(h)        # linear unembedding H_{L+1} → O

    No residual connections. The cloning operator is implicit
    (input is fed to both branches of the bilinear core).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # +1 for appended constant (bias trick)
        self.embed = nn.Linear(input_dim + 1, hidden_dim, bias=False)
        self.cores = nn.ModuleList([
            BilinearCore(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            RMSBatchNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # Append constant 1 for bias
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        x = torch.cat([ones, x], dim=-1)

        h = self.embed(x)
        for norm, core in zip(self.norms, self.cores):
            h = norm(h)
            h = core(h)
        return self.unembed(h)

    def get_cores_for_odt(self):
        """
        Extract the core tensors for ODT analysis.

        After training, fold the RMSBatchNorm running averages into
        adjacent weights, then return:
        - embed_matrix: (hidden_dim, input_dim+1)
        - core_tensors: list of (hidden_dim, hidden_dim, hidden_dim) tensors
        - unembed_matrix: (output_dim, hidden_dim)
        """
        # Fold norms into embedding/cores
        # norm divides by running_rms, so absorb 1/rms into the
        # preceding layer's output weights.

        # Start with embed, absorb norm[0]
        embed_w = self.embed.weight.clone()  # (h, input+1)
        scale = 1.0 / (self.norms[0].running_rms + self.norms[0].eps)
        embed_w = embed_w * scale  # scale the output of embed

        core_tensors = []
        for i, core in enumerate(self.cores):
            A = core.A.weight.clone()  # (h, h)
            B = core.B.weight.clone()  # (h, h)

            # If there's a next norm, absorb it into this core's output
            if i + 1 < self.n_layers:
                next_scale = 1.0 / (self.norms[i + 1].running_rms + self.norms[i + 1].eps)
                A_scaled = A * next_scale.unsqueeze(-1) if next_scale.dim() > 0 else A * next_scale
                B_scaled = B  # Only need to scale one of A or B? No...
                # Actually the output is A(h)*B(h), and norm scales the output.
                # So we need to scale the full output by next_scale.
                # Output_o = sum_i A_{oi} h_i * sum_j B_{oj} h_j
                # Scaling output: next_scale * Output_o
                # Can absorb into A: A_{oi} -> next_scale * A_{oi}
                A = A * next_scale

            # Build symmetrized interaction tensor
            T = torch.einsum('oi,oj->oij', A, B)
            T = 0.5 * (T + T.transpose(1, 2))
            core_tensors.append(T)

        unembed_w = self.unembed.weight.clone()  # (output, h)

        return embed_w, core_tensors, unembed_w


class ResidualBilinearNet(nn.Module):
    """
    Bilinear network WITH residual connections.

    Architecture:
        x ← (1, x)
        h = embed(x)
        for each layer:
            h = h + bilinear(norm(h))   # RESIDUAL
        y = unembed(h)

    This is the practical architecture. For analysis, the residual
    can be folded into the core by augmenting the bilinear tensor
    with an identity component.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embed = nn.Linear(input_dim + 1, hidden_dim, bias=False)
        self.cores = nn.ModuleList([
            BilinearCore(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            RMSBatchNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        x = torch.cat([ones, x], dim=-1)

        h = self.embed(x)
        for norm, core in zip(self.norms, self.cores):
            h = h + core(norm(h))
        return self.unembed(h)

    def get_cores_for_odt(self, fold_residual=False):
        """
        Extract cores for ODT analysis (bilinear part only).

        Returns the bilinear interaction tensors with norms folded in.
        This loses the residual path information — for full residual
        analysis, use get_augmented_cores_for_odt() instead.

        Note: fold_residual=True is DEPRECATED and incorrect. It assumed
        h[0]=1 in hidden space, which is not guaranteed after the embedding.
        Use get_augmented_cores_for_odt() for correct residual folding.
        """
        if fold_residual:
            import warnings
            warnings.warn(
                "fold_residual=True is incorrect (assumes h[0]=1 in hidden space). "
                "Use get_augmented_cores_for_odt() instead.",
                DeprecationWarning, stacklevel=2
            )

        embed_w = self.embed.weight.clone()
        scale = 1.0 / (self.norms[0].running_rms + self.norms[0].eps)
        embed_w = embed_w * scale

        core_tensors = []
        for i, core in enumerate(self.cores):
            A = core.A.weight.clone()
            B = core.B.weight.clone()

            if i + 1 < self.n_layers:
                next_scale = 1.0 / (self.norms[i + 1].running_rms + self.norms[i + 1].eps)
                A = A * next_scale

            T = torch.einsum('oi,oj->oij', A, B)
            T = 0.5 * (T + T.transpose(1, 2))
            core_tensors.append(T)

        unembed_w = self.unembed.weight.clone()
        return embed_w, core_tensors, unembed_w

    def get_augmented_cores_for_odt(self):
        """
        Extract augmented cores for ODT analysis with correct residual folding.

        The residual h_out = h + bilinear(norm(h)) is represented by augmenting
        the hidden space with a constant dimension: ĥ = [1, h] (dim H+1).

        Each augmented core C of shape (H+1, H+1, H+1) computes:
            ĥ_out[0] = 1  (constant preservation)
            ĥ_out[o+1] = alpha * g[o] + bilinear_folded(g)[o]
        where g = ĥ[1:] is the hidden state and alpha accounts for
        normalization rescaling between layers.

        The "discard trick" from the TN literature: the residual (degree 1)
        is lifted to degree 2 by contracting one leg against the constant
        dimension (index 0). The bilinear part (degree 2) occupies indices ≥1.

        Norms are folded into weights:
        - First norm (1/rms_0) absorbed into embedding
        - Subsequent norms (1/rms_{i+1}) absorbed into core i's bilinear output
        - Residual coefficient: alpha_i = rms_i / rms_{i+1} (intermediate),
          alpha_{L-1} = rms_{L-1} (last layer)

        Returns:
            embed_aug: (H+1, input+1) augmented embedding
            cores_aug: list of (H+1, H+1, H+1) augmented core tensors
            unembed_aug: (output, H+1) augmented unembedding
        """
        H = self.hidden_dim
        L = self.n_layers

        # --- Augmented embedding ---
        # Fold first norm: g_0 = embed(x) / rms_0
        embed_w = self.embed.weight.clone()  # (H, input+1)
        rms_0 = (self.norms[0].running_rms + self.norms[0].eps).item()
        embed_w = embed_w / rms_0

        # Augment: [1, g_0] = embed_aug @ x_aug
        embed_aug = torch.zeros(H + 1, embed_w.shape[1])
        embed_aug[0, 0] = 1.0   # constant maps to constant (x_aug[0] = 1)
        embed_aug[1:, :] = embed_w

        # --- Augmented cores ---
        cores_aug = []
        for i in range(L):
            A = self.cores[i].A.weight.clone()  # (H, H)
            B = self.cores[i].B.weight.clone()  # (H, H)

            rms_i = (self.norms[i].running_rms + self.norms[i].eps).item()

            if i + 1 < L:
                rms_next = (self.norms[i + 1].running_rms + self.norms[i + 1].eps).item()
                A = A / rms_next   # fold next norm into bilinear
                alpha = rms_i / rms_next   # residual coefficient
            else:
                alpha = rms_i   # last layer: h_L = rms_{L-1} * g + bilinear(g)

            # Symmetrized bilinear tensor
            T = torch.einsum('oi,oj->oij', A, B)
            T = 0.5 * (T + T.transpose(1, 2))

            # Build augmented core (H+1, H+1, H+1)
            C = torch.zeros(H + 1, H + 1, H + 1)
            C[0, 0, 0] = 1.0           # constant preservation
            C[1:, 1:, 1:] = T          # bilinear part (indices ≥ 1)

            # Residual via discard trick: identity * alpha, symmetrized
            idx = torch.arange(1, H + 1)
            C[idx, idx, 0] = alpha / 2
            C[idx, 0, idx] = alpha / 2

            cores_aug.append(C)

        # --- Augmented unembedding ---
        unembed_w = self.unembed.weight.clone()  # (output, H)
        unembed_aug = torch.zeros(unembed_w.shape[0], H + 1)
        unembed_aug[:, 1:] = unembed_w   # constant dim doesn't affect output

        return embed_aug, cores_aug, unembed_aug
