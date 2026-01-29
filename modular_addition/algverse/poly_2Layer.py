import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


class BilinearStack(nn.Module):
    """
    Stack of bilinear layers with skip connections.
    
    Each layer: h = h + D(Lh ⊙ Rh)   [bilinear only]
           or:  h = h + D(Lh ⊙ Rh) + Wh  [with linear]
    
    Args:
        n: sequence length / input dim
        num_layers: number of bilinear layers
        rank: rank of each bilinear layer
        use_linear: whether to add W matrix at each layer
    """
    def __init__(self, n, num_layers=2, rank=None, use_linear=False):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.rank = rank if rank is not None else n
        self.use_linear = use_linear
        
        # Build layers
        self.Ls = nn.ParameterList()
        self.Rs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        self.Ws = nn.ParameterList() if use_linear else None
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.Ls.append(nn.Parameter(torch.randn(self.rank, n) * 0.1))
            self.Rs.append(nn.Parameter(torch.randn(self.rank, n) * 0.1))
            self.Ds.append(nn.Parameter(torch.randn(n, self.rank) * 0.1))
            if use_linear:
                self.Ws.append(nn.Parameter(torch.randn(n, n) * 0.1))
            self.norms.append(RMSNorm(n))
    
    def forward(self, x):
        # x: (batch, n)
        h = x
        for i in range(self.num_layers):
            h = self.norms[i](h)
            Lh = h @ self.Ls[i].T  # (batch, rank)
            Rh = h @ self.Rs[i].T  # (batch, rank)
            bilinear = (Lh * Rh) @ self.Ds[i].T  # (batch, n)
            
            if self.use_linear:
                linear = h @ self.Ws[i].T
                h = h + bilinear + linear
            else:
                h = h + bilinear
            
        
        return h
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def max_degree(self):
        """Maximum polynomial degree this architecture can represent."""
        return 2 ** self.num_layers


def train_and_test(model, n, steps=10000, lr=0.01, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for step in range(steps):
        x = torch.randn(512, n)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and step % 2000 == 0:
            acc = (logits.argmax(-1) == targets).float().mean()
            print(f"Step {step}: loss={loss.item():.3f}, acc={acc.item():.3f}")
    
    # Final test
    model.eval()
    with torch.no_grad():
        x = torch.randn(100000, n)
        targets = task_2nd_argmax(x)
        logits = model(x)
        acc = (logits.argmax(-1) == targets).float().mean().item()
    
    return acc


def run_experiments():
    """Test different configurations."""
    
    results = []
    
    for n in [10]:
        print(f"\n{'='*60}")
        print(f"n = {n} (need degree {n-1}, random baseline = {1/n:.1%})")
        print('='*60)
        
        for num_layers in [2,3,4]:
            for rank in [5, 10, 32, 128]:
                for use_linear in [True]:
                    model = BilinearStack(
                        n=n, 
                        num_layers=num_layers, 
                        rank=rank,
                        # rank=n*n*n,

                        use_linear=use_linear
                    )
                    
                    desc = f"layers={num_layers}, rank={rank}, linear={use_linear}"
                    params = model.count_params()
                    max_deg = model.max_degree()
                    
                    print(f"\n{desc}")
                    print(f"  params={params}, max_degree={max_deg}")
                    
                    acc = train_and_test(model, n, steps=20_000, verbose=True)
                    print(f"  accuracy: {acc:.1%}")
                    
                    results.append({
                        'n': n,
                        'num_layers': num_layers,
                        'use_linear': use_linear,
                        'params': params,
                        'max_degree': max_deg,
                        'accuracy': acc
                    })
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{'n':>3} {'layers':>6} {'linear':>6} {'params':>6} {'degree':>6} {'acc':>8}")
    print("-"*45)
    for r in results:
        print(f"{r['n']:>3} {r['num_layers']:>6} {str(r['use_linear']):>6} "
              f"{r['params']:>6} {r['max_degree']:>6} {r['accuracy']:>7.1%}")
    
    return results


if __name__ == "__main__":
    # Quick test for n=5
    # print("Quick test: n=5, 2 layers, with linear")
    # model = BilinearStack(n=5, num_layers=2, rank=5, use_linear=True)
    # print(f"Params: {model.count_params()}, Max degree: {model.max_degree()}")
    # acc = train_and_test(model, n=5, steps=10000, verbose=True)
    # print(f"Final: {acc:.1%}")
    # run_experiments
    # Uncomment to run full experiments:
    results = run_experiments()
# ```

# ## Architecture Summary
# ```
# BilinearStack(n, num_layers, rank, use_linear)

# Each layer:
#   h = h + D(Lh ⊙ Rh)           # use_linear=False
#   h = h + D(Lh ⊙ Rh) + Wh      # use_linear=True

# Parameters per layer:
#   - Bilinear only:  3 * n * rank
#   - With linear:    3 * n * rank + n²

# Max polynomial degree: 2^num_layers