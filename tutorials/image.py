import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import *

from torch.utils.data import Dataset
from torchvision import datasets

class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)
        
from shared import Linear, Bilinear

def _collator(transform=None):
    def inner(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return (x, y) if transform is None else (transform(x), y)
    return inner

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        n_layer: int = 1,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        residual: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
    
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.residual = residual
        
        
        
        super().__init__(**kwargs)


class Model(PreTrainedModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        d_input, d_hidden, d_output = config.d_input, config.d_hidden, config.d_output
        bias, n_layer = config.bias, config.n_layer
        
        self.embed = Linear(d_input, d_hidden, bias=False)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x.flatten(start_dim=1))
        
        for layer in self.blocks:
            x = x + layer(x) if self.config.residual else layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_lr(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @property
    def w_l(self):
        return self.w_lr.unbind(1)[0]
    
    @property
    def w_r(self):
        return self.w_lr.unbind(1)[1]
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy
    
    def fit(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_x, test_y = test.x, test.y
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

    def decompose(self):
        """The function to decompose a single-layer model into eigenvalues and eigenvectors."""
        
        # Split the bilinear layer into the left and right components
        l, r = self.w_lr[0].unbind()
        
        # Compute the third-order (bilinear) tensor
        b = einsum(self.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
        
        # Symmetrize the tensor
        b = 0.5 * (b + b.mT)

        # Perform the eigendecomposition
        vals, vecs = torch.linalg.eigh(b)
        
        # Project the eigenvectors back to the input space
        vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
        
        # Return the eigenvalues and eigenvectors
        return vals, vecs
    

from jaxtyping import Float
from torch import Tensor
from einops import *
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.express as px
import torch

def plot_explanation(model, sample: Float[Tensor, "w h"], eigenvalues=10):
    """Creates a plot showing the top eigenvector activations for a given input sample."""
    colors = px.colors.qualitative.Plotly
    
    logits = model(sample)[0].cpu()
    classes = logits.topk(3).indices.sort().values.cpu()
    
    # compute the activations of the eigenvectors for a given sample
    vals, vecs = model.decompose()
    vals, vecs = vals.cpu(), vecs.cpu()
    acts = einsum(sample.flatten().cpu(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals

    # compute the contributions of the top 3 classes
    contrib, idxs = acts[classes].sort(dim=-1)

    titles = [''] + [f"{c}" for c in classes] + ['input', ''] + [f"{c}" for c in classes] + ['logits']
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles, vertical_spacing=0.1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    
    # add line plot for eigenvalues
    for i in range(3):
        params = dict(showlegend=False, marker=dict(color=colors[i]))
        fig.add_scatter(y=contrib[i, -eigenvalues-2:].flip(0), mode="lines", **params, row=1, col=1)
        fig.add_scatter(y=contrib[i, -1:].flip(0), mode="markers", **params, row=1, col=1)
        
        fig.add_scatter(y=contrib[i, :eigenvalues+2], mode="lines", **params, row=2, col=1)
        fig.add_scatter(y=contrib[i, :1], mode="markers", **params, row=2, col=1)
    
    # add heatmaps for the top 3 classes
    for i in range(3):
        params = dict(showscale=False, colorscale="RdBu", zmid=0)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, -1]].view(28, 28).flip(0), **params, row=1, col=i+2)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, 0]].view(28, 28).flip(0), **params, row=2, col=i+2)
    
    # add tickmarks for the heatmaps
    for i in range(2):
        tickvals = [0] + list(contrib[:3, [-1, 0][i]])
        ticktext = [f'{val:.2f}' for val in tickvals]
        fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=i+1)
    
    bars, text = ["gray"] * 10, [""] * 10
    for i, c in zip(classes, colors):
        bars[i], text[i] = c, f"{i}"

    fig.add_bar(y=logits, marker_color=bars, text=text, showlegend=False, textposition='outside', textfont=dict(size=12), row=2, col=5)
    fig.update_yaxes(range=[logits.min(), logits.max() * 1.5], row=2, col=5)
    
    fig.add_heatmap(z=sample[0].flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=5)
    fig.update_annotations(font_size=13)
    
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_layout(width=800, height=320, margin=dict(l=0, r=0, b=0, t=20), template="plotly_white")

    return fig


def plot_eigenspectrum(model, digit, eigenvectors=3, eigenvalues=20, ignore_pos=[], ignore_neg=[]):
    """Plot the eigenspectrum for a given digit."""
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1 + eigenvectors)
    
    vals, vecs = model.decompose()
    vals, vecs = vals[digit].cpu(), vecs[digit].cpu()
    
    negative = torch.arange(eigenvectors)
    positive = -1 - negative

    fig.add_trace(go.Scatter(y=vals[-eigenvalues-2:].flip(0), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=negative.flip(0), y=vals[positive].flip(0), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    fig.add_trace(go.Scatter(y=vals[:eigenvalues+2], mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative, y=vals[negative], mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    for i, idx in enumerate(positive):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1)
    
    tickvals = [0] + [x.item() for i, x in enumerate(vals[positive]) if i not in ignore_pos]
    ticktext = [f'{val:.2f}' for val in tickvals]
    
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [x.item() for i, x in enumerate(vals[negative]) if i not in ignore_neg]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_coloraxes(showscale=False)
    fig.update_layout(autosize=False, width=170*(eigenvectors+1), height=300, margin=dict(l=0, r=0, b=0, t=0), template="plotly_white")
    fig.update_legends(visible=False)
    
    return fig