"""
Interactive Class Circuit Viewer for Sparse Bilinear Networks.

Computes variance-based backward tracing per class and outputs a
self-contained HTML file with embedded thumbnails and circuit graph.

Usage:
    cd /workspace/tn_4_interp/odt && python ../sparse_circuits/build_class_viewer.py
"""

import sys, os, json, base64, io, time

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

from train import get_svhn_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
DATA_ROOT = os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data")
MAX_SAMPLES = 5000
CHECKPOINT = os.path.join(RESULTS_DIR, "chi_net_L3_h256_folded_sparse.pt")
OUTPUT_HTML = os.path.join(RESULTS_DIR, "class_circuit_viewer.html")


# ================================================================
# Step 1: Load model + compute activations
# ================================================================

def load_sparse_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


@torch.no_grad()
def compute_activations(cores, embed, data_loader, device, max_samples=5000):
    """Forward pass, record activations at every bond."""
    embed_d = embed.to(device)
    cores_d = [c.to(device) for c in cores]

    n_bonds = len(cores) + 1
    all_acts = [[] for _ in range(n_bonds)]
    all_labels = []
    all_images = []
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        ones = torch.ones(x.shape[0], 1, device=device)
        h = torch.cat([ones, x], dim=-1)
        h = h @ embed_d.T
        all_acts[0].append(h.cpu())

        for i, core in enumerate(cores_d):
            h = torch.einsum('oij,bi,bj->bo', core, h, h)
            all_acts[i + 1].append(h.cpu())

        all_labels.append(y.cpu())
        all_images.append(x.cpu())
        total += x.shape[0]
        if total >= max_samples:
            break

    activations = [torch.cat(a, dim=0).numpy()[:max_samples] for a in all_acts]
    labels = torch.cat(all_labels).numpy()[:max_samples]
    images = torch.cat(all_images).numpy()[:max_samples]
    return activations, labels, images


# ================================================================
# Step 2: Variance-based backward trace per class
# ================================================================

def trace_class_circuit(cores, activations, class_idx, var_threshold=0.99):
    """Backward trace from class_idx at Bond 3 through all layers.

    Returns dict with per-layer edges and selected features.
    """
    n_layers = len(cores)
    circuit = {
        'class': class_idx,
        'layers': [],  # one entry per layer, from last to first
        'selected_features': {},  # bond -> list of feature indices
    }

    # Start: Bond 3 targets = {class_idx}
    targets = {class_idx}
    circuit['selected_features'][n_layers] = sorted(targets)

    for layer_idx in range(n_layers - 1, -1, -1):
        core = cores[layer_idx]  # (r_out, r_in, r_in)
        T = 0.5 * (core + core.transpose(-1, -2))  # symmetrize
        h_in = activations[layer_idx]  # (n_samples, r_in)
        r_in = T.shape[1]

        layer_edges = []
        input_features = set()

        for k in sorted(targets):
            T_k = T[k].numpy()  # (r_in, r_in)
            edges_for_k = []

            for i in range(r_in):
                for j in range(i, r_in):
                    val = T_k[i, j]
                    if i != j:
                        val = T_k[i, j] + T_k[j, i]
                    if abs(val) < 1e-10:
                        continue

                    w = val
                    contribution = w * h_in[:, i] * h_in[:, j]
                    var = float(np.var(contribution))
                    if var < 1e-15:
                        continue

                    edges_for_k.append({
                        'src_i': int(i),
                        'src_j': int(j),
                        'dst_k': int(k),
                        'weight': float(w),
                        'variance': var,
                    })

            # Sort by variance descending, keep up to threshold
            edges_for_k.sort(key=lambda e: e['variance'], reverse=True)
            total_var = sum(e['variance'] for e in edges_for_k)

            if total_var > 0:
                cum_var = 0.0
                for e in edges_for_k:
                    cum_var += e['variance']
                    e['cum_var_pct'] = cum_var / total_var
                    layer_edges.append(e)
                    input_features.add(e['src_i'])
                    input_features.add(e['src_j'])
                    if cum_var / total_var >= var_threshold:
                        break

        circuit['layers'].insert(0, {
            'layer_idx': layer_idx,
            'edges': layer_edges,
        })
        circuit['selected_features'][layer_idx] = sorted(input_features)
        targets = input_features

    return circuit


def compute_importance_for_circuit(circuit, threshold=0.99):
    """Compute per-feature importance scores for a traced circuit.

    Returns: {bond: {feat: importance_score}}
    """
    n_layers = len(circuit['layers'])
    n_bonds = n_layers + 1
    last_bond = max(int(k) for k in circuit['selected_features'].keys())

    importance = {last_bond: {circuit['class']: 1.0}}

    for layer_data in reversed(circuit['layers']):
        layer_idx = layer_data['layer_idx']
        bond_out = layer_idx + 1
        bond_in = layer_idx

        feat_imp = {}

        by_dst = {}
        for e in layer_data['edges']:
            dk = e['dst_k']
            if dk not in by_dst:
                by_dst[dk] = []
            by_dst[dk].append(e)

        for dk, edges in by_dst.items():
            target_imp = importance.get(bond_out, {}).get(dk)
            if target_imp is None:
                continue
            total_var = sum(e['variance'] for e in edges)
            if total_var == 0:
                continue

            for e in edges:
                if e['cum_var_pct'] <= threshold:
                    edge_imp = target_imp * e['variance'] / total_var
                    feat_imp[e['src_i']] = feat_imp.get(e['src_i'], 0) + edge_imp
                    if e['src_i'] != e['src_j']:
                        feat_imp[e['src_j']] = feat_imp.get(e['src_j'], 0) + edge_imp

        importance[bond_in] = feat_imp

    return importance


# ================================================================
# Step 3: Feature metadata + thumbnails
# ================================================================

def trace_feature_to_input(cores, embed, bond, feature_idx):
    """Trace a feature backward to pixel space via dominant eigenvector."""
    w = torch.zeros(cores[bond - 1].shape[0] if bond > 0 else embed.shape[0])
    w[feature_idx] = 1.0

    for layer_idx in range(bond - 1, -1, -1):
        core = cores[layer_idx]
        M = torch.einsum('o,oij->ij', w, core)
        M = 0.5 * (M + M.T)
        try:
            evals, evecs = torch.linalg.eigh(M)
            idx = evals.abs().argsort(descending=True)[0]
            w = evecs[:, idx]
        except torch._C._LinAlgError:
            U, S, _ = torch.linalg.svd(M)
            w = U[:, 0]

    input_pattern = embed.T @ w
    input_pattern = input_pattern[1:]  # skip constant
    n_pixels = 32 * 32
    return input_pattern[:n_pixels].reshape(32, 32).detach().numpy()


def make_thumbnail_b64(arr, size=64):
    """Convert a 2D array to a base64-encoded PNG thumbnail."""
    vmax = np.abs(arr).max()
    if vmax < 1e-10:
        vmax = 1.0
    normalized = (arr / vmax + 1) / 2  # map [-vmax,vmax] to [0,1]
    colored = (cm.RdBu_r(normalized) * 255).astype(np.uint8)
    img = Image.fromarray(colored[:, :, :3]).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ================================================================
# Step 4: Generate HTML
# ================================================================

def generate_html(circuits, feature_meta, ranks, acc, sparsity_pct):
    """Generate self-contained HTML file."""

    # Convert circuits to JSON-serializable format
    circuits_json = []
    for circ in circuits:
        c = {
            'class': circ['class'],
            'layers': circ['layers'],
            'selected_features': {str(k): v for k, v in circ['selected_features'].items()},
        }
        circuits_json.append(c)

    # Convert feature metadata to JSON (keyed by "bond_feat")
    meta_json = {}
    for (bond, feat), m in feature_meta.items():
        key = f"{bond}_{feat}"
        meta_json[key] = m

    data_json = json.dumps({
        'circuits': circuits_json,
        'features': meta_json,
        'ranks': ranks,
        'acc': acc,
        'sparsity_pct': sparsity_pct,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sparse Bilinear Circuit Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #e6e6e6; font-family: 'Segoe UI', system-ui, sans-serif; overflow-x: hidden; }}
#header {{ padding: 16px 24px; background: #16213e; border-bottom: 2px solid #0f3460; }}
#header h1 {{ font-size: 20px; font-weight: 600; color: #e94560; }}
#header .subtitle {{ font-size: 13px; color: #999; margin-top: 4px; }}
#controls {{ display: flex; align-items: center; gap: 16px; padding: 12px 24px; background: #16213e; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }}
.class-tabs {{ display: flex; gap: 4px; }}
.class-tab {{ padding: 6px 14px; border: 1px solid #0f3460; border-radius: 6px; background: #1a1a2e; color: #e6e6e6; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.15s; }}
.class-tab:hover {{ background: #0f3460; }}
.class-tab.active {{ background: #e94560; border-color: #e94560; color: white; }}
.slider-group {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
.slider-group label {{ font-size: 13px; color: #999; }}
#threshold-slider {{ width: 200px; accent-color: #e94560; }}
#threshold-val {{ font-size: 14px; font-weight: 600; color: #e94560; min-width: 40px; }}
#main {{ display: flex; position: relative; min-height: calc(100vh - 120px); }}
#circuit-container {{ flex: 1; display: flex; justify-content: space-around; position: relative; padding: 24px 40px; overflow: auto; gap: 20px; }}
.bond-column {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 180px; max-width: 280px; padding: 0 8px; flex-shrink: 0; }}
.bond-header {{ font-size: 13px; color: #999; margin-bottom: 12px; text-align: center; }}
.bond-header .dim {{ font-size: 11px; color: #666; }}
.feature-node {{ position: relative; background: #0f3460; border: 1px solid #1a3a6e; border-radius: 8px; padding: 8px; margin: 5px 0; cursor: pointer; transition: all 0.15s; width: 170px; text-align: center; }}
.feature-node:hover {{ border-color: #e94560; box-shadow: 0 0 12px rgba(233,69,96,0.4); z-index: 10; }}
.feature-node.highlighted {{ border-color: #e94560; box-shadow: 0 0 12px rgba(233,69,96,0.4); z-index: 10; }}
.feature-node.dimmed {{ opacity: 0.2; }}
.feature-node .thumb {{ width: 72px; height: 72px; border-radius: 4px; margin: 0 auto 4px; display: block; image-rendering: pixelated; }}
.feature-node .feat-label {{ font-size: 12px; font-weight: 600; color: #e6e6e6; }}
.feature-node .feat-stats {{ font-size: 9px; color: #888; margin-top: 2px; }}
.feature-node .feat-imp {{ font-size: 9px; color: #e94560; margin-top: 1px; }}
.mini-bars {{ display: flex; gap: 1px; justify-content: center; margin-top: 3px; height: 14px; align-items: flex-end; }}
.mini-bar {{ width: 7px; border-radius: 1px 1px 0 0; min-height: 1px; }}
.class-node {{ background: #2a1a3e; border-color: #e94560; }}
.class-node .feat-label {{ color: #e94560; font-size: 14px; }}
svg#edges {{ position: absolute; top: 0; left: 0; pointer-events: none; z-index: 5; }}
svg#edges path {{ transition: opacity 0.15s; }}
#modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }}
#modal-overlay.visible {{ display: flex; }}
#modal-content {{ background: #16213e; border-radius: 12px; padding: 20px; max-width: 90vw; max-height: 90vh; overflow: auto; position: relative; }}
#modal-content .close-btn {{ position: absolute; top: 8px; right: 12px; font-size: 24px; color: #999; cursor: pointer; background: none; border: none; z-index: 10; }}
#modal-content .close-btn:hover {{ color: #e94560; }}
#modal-content img {{ max-width: 100%; border-radius: 4px; }}
#modal-content .modal-info {{ margin-top: 12px; }}
#modal-content .modal-info h3 {{ color: #e94560; margin-bottom: 8px; }}
#modal-content .modal-info p {{ font-size: 13px; color: #ccc; line-height: 1.5; }}
.sel-bar {{ display: flex; gap: 1px; margin-top: 6px; }}
.sel-bar-item {{ width: 20px; text-align: center; font-size: 8px; }}
.sel-bar-fill {{ height: 24px; border-radius: 2px; }}
#legend {{ position: fixed; bottom: 12px; right: 12px; background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 10px 14px; font-size: 11px; color: #999; z-index: 50; }}
#legend .leg-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
#legend .leg-swatch {{ width: 20px; height: 3px; border-radius: 2px; }}
</style>
</head>
<body>

<div id="header">
  <h1>Sparse Bilinear Circuit Viewer</h1>
  <div class="subtitle">chi_net_L3_h256 | {acc:.1f}% acc | {sparsity_pct:.0f}% sparse | ranks {ranks}</div>
</div>

<div id="controls">
  <div class="class-tabs" id="class-tabs"></div>
  <div class="slider-group">
    <label>Variance explained:</label>
    <input type="range" id="threshold-slider" min="10" max="99" value="90" step="1">
    <span id="threshold-val">90%</span>
  </div>
</div>

<div id="main">
  <div id="circuit-container"></div>
  <svg id="edges"></svg>
</div>

<div id="modal-overlay" onclick="closeModal(event)">
  <div id="modal-content">
    <button class="close-btn" onclick="closeModal(event)">&times;</button>
    <div id="modal-body"></div>
  </div>
</div>

<div id="legend">
  <div class="leg-item"><div class="leg-swatch" style="background:#e94560"></div> Excitatory (w &gt; 0)</div>
  <div class="leg-item"><div class="leg-swatch" style="background:#4361ee"></div> Inhibitory (w &lt; 0)</div>
  <div class="leg-item"><div class="leg-swatch" style="background:#e94560;border-top:2px dashed #e94560;height:0"></div> Dashed = cross-term (i&ne;j)</div>
  <div class="leg-item">Width &prop; variance contribution</div>
  <div class="leg-item">Click node for details</div>
</div>

<script>
const DATA = {data_json};

let currentClass = 0;
let currentThreshold = 0.90;
const MIN_SOURCES = 2;  // min unique source features per target

const CLASS_COLORS = [
  '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
];

function init() {{
  const tabsEl = document.getElementById('class-tabs');
  for (let c = 0; c < 10; c++) {{
    const tab = document.createElement('div');
    tab.className = 'class-tab' + (c === 0 ? ' active' : '');
    tab.textContent = c;
    tab.onclick = () => selectClass(c);
    tab.id = 'tab-' + c;
    tabsEl.appendChild(tab);
  }}

  const slider = document.getElementById('threshold-slider');
  const valEl = document.getElementById('threshold-val');
  slider.oninput = () => {{
    currentThreshold = parseInt(slider.value) / 100;
    valEl.textContent = slider.value + '%';
    renderCircuit();
  }};

  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') closeModal(e);
    if (e.key >= '0' && e.key <= '9') selectClass(parseInt(e.key));
  }});

  const ro = new ResizeObserver(() => renderEdges());
  ro.observe(document.getElementById('circuit-container'));

  renderCircuit();
}}

function selectClass(idx) {{
  currentClass = idx;
  document.querySelectorAll('.class-tab').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});
  renderCircuit();
}}

function computeVisibleFeatures(circuit, threshold) {{
  // Backward pass: filter edges by threshold (with min-source guarantee),
  // compute recursive importance, sort by class-specificity.
  const nBonds = DATA.ranks.length;
  const result = {{
    features: {{}},     // bond -> [feat, ...] ordered by class-specific importance
    edges: [],
    importance: {{}},    // bond -> {{feat: score}}
    specificity: {{}}    // bond -> {{feat: imp - mean_imp}}
  }};

  // Bond 3: class node
  result.features[nBonds - 1] = [circuit.class];
  result.importance[nBonds - 1] = {{}};
  result.importance[nBonds - 1][circuit.class] = 1.0;

  // Process layers backward
  for (let li = circuit.layers.length - 1; li >= 0; li--) {{
    const layer = circuit.layers[li];
    const layerIdx = layer.layer_idx;
    const bondOut = layerIdx + 1;
    const bondIn = layerIdx;

    const featureImp = {{}};

    // Group edges by dst_k
    const byDst = {{}};
    for (const e of layer.edges) {{
      if (!byDst[e.dst_k]) byDst[e.dst_k] = [];
      byDst[e.dst_k].push(e);
    }}

    for (const [dstK, edges] of Object.entries(byDst)) {{
      const dk = parseInt(dstK);
      const targetImp = result.importance[bondOut]?.[dk];
      if (targetImp === undefined) continue;

      const totalVar = edges.reduce((s, e) => s + e.variance, 0);
      if (totalVar === 0) continue;

      // Walk edges (sorted by variance desc from Python).
      // Include if below threshold OR if we haven't reached MIN_SOURCES yet.
      const uniqueSources = new Set();
      for (const e of edges) {{
        const belowThreshold = e.cum_var_pct <= threshold;
        const needMore = uniqueSources.size < MIN_SOURCES;

        if (!belowThreshold && !needMore) break;

        result.edges.push({{ ...e, layer_idx: layerIdx }});
        uniqueSources.add(e.src_i);
        if (e.src_i !== e.src_j) uniqueSources.add(e.src_j);

        const edgeImp = targetImp * e.variance / totalVar;
        featureImp[e.src_i] = (featureImp[e.src_i] || 0) + edgeImp;
        if (e.src_i !== e.src_j) {{
          featureImp[e.src_j] = (featureImp[e.src_j] || 0) + edgeImp;
        }}
      }}
    }}

    result.importance[bondIn] = featureImp;

    // Sort by class-specific importance: imp - mean_imp
    // mean_imp is precomputed in Python across all 10 classes
    const specificity = {{}};
    for (const [feat, imp] of Object.entries(featureImp)) {{
      const meanImp = DATA.features[bondIn + '_' + feat]?.mean_importance || 0;
      specificity[feat] = imp - meanImp;
    }}
    result.specificity[bondIn] = specificity;

    result.features[bondIn] = Object.keys(featureImp)
      .map(Number)
      .sort((a, b) => (specificity[b] || 0) - (specificity[a] || 0));
  }}

  return result;
}}

function renderCircuit() {{
  const circuit = DATA.circuits[currentClass];
  const visible = computeVisibleFeatures(circuit, currentThreshold);
  const container = document.getElementById('circuit-container');
  container.innerHTML = '';

  const nBonds = DATA.ranks.length;

  for (let bond = 0; bond < nBonds; bond++) {{
    const col = document.createElement('div');
    col.className = 'bond-column';
    col.id = 'bond-col-' + bond;

    const header = document.createElement('div');
    header.className = 'bond-header';
    const feats = visible.features[bond] || [];
    header.innerHTML = `Bond ${{bond}}<br><span class="dim">(${{DATA.ranks[bond]}} dim, ${{feats.length}} shown)</span>`;
    col.appendChild(header);

    for (const feat of feats) {{
      const key = bond + '_' + feat;
      const meta = DATA.features[key];
      const node = document.createElement('div');
      const isClassNode = (bond === nBonds - 1);
      node.className = 'feature-node' + (isClassNode ? ' class-node' : '');
      node.id = 'node-' + bond + '-' + feat;
      node.dataset.bond = bond;
      node.dataset.feat = feat;

      let label, statsHtml, impHtml;
      if (isClassNode) {{
        label = 'Class ' + feat;
        statsHtml = '';
        impHtml = '';
      }} else {{
        label = 'f' + feat;
        if (meta) {{
          statsHtml = `<div class="feat-stats">var=${{meta.variance.toFixed(3)}} | +${{(meta.frac_pos*100).toFixed(0)}}% -${{(meta.frac_neg*100).toFixed(0)}}%</div>`;
        }} else {{
          statsHtml = '';
        }}
        const spec = visible.specificity[bond]?.[feat];
        const imp = visible.importance[bond]?.[feat];
        if (spec !== undefined && imp !== undefined) {{
          impHtml = `<div class="feat-imp">spec=${{spec.toFixed(3)}} (imp=${{imp.toFixed(3)}})</div>`;
        }} else {{
          impHtml = '';
        }}
      }}

      let thumbHtml = '';
      if (meta && meta.thumbnail_b64) {{
        thumbHtml = `<img class="thumb" src="data:image/png;base64,${{meta.thumbnail_b64}}" alt="f${{feat}}">`;
      }}

      // Mini class selectivity bars
      let barsHtml = '';
      if (meta && meta.class_selectivity) {{
        const maxSel = Math.max(...meta.class_selectivity.map(Math.abs), 0.01);
        barsHtml = '<div class="mini-bars">';
        for (let c = 0; c < 10; c++) {{
          const v = meta.class_selectivity[c];
          const h = Math.round(Math.abs(v) / maxSel * 14);
          const color = v >= 0 ? CLASS_COLORS[c] : '#666';
          barsHtml += `<div class="mini-bar" style="height:${{Math.max(1,h)}}px;background:${{color}}" title="class ${{c}}: ${{v.toFixed(3)}}"></div>`;
        }}
        barsHtml += '</div>';
      }}

      node.innerHTML = thumbHtml + `<div class="feat-label">${{label}}</div>` + statsHtml + impHtml + barsHtml;

      node.onclick = (e) => {{ e.stopPropagation(); showFeatureModal(bond, feat); }};
      node.onmouseenter = () => highlightEdges(bond, feat);
      node.onmouseleave = () => clearHighlights();

      col.appendChild(node);
    }}

    container.appendChild(col);
  }}

  requestAnimationFrame(() => requestAnimationFrame(() => renderEdges()));
}}

function renderEdges() {{
  const circuit = DATA.circuits[currentClass];
  const visible = computeVisibleFeatures(circuit, currentThreshold);
  const svg = document.getElementById('edges');
  const container = document.getElementById('circuit-container');
  const rect = container.getBoundingClientRect();

  svg.setAttribute('width', container.scrollWidth);
  svg.setAttribute('height', container.scrollHeight);
  svg.style.width = container.scrollWidth + 'px';
  svg.style.height = container.scrollHeight + 'px';
  svg.innerHTML = '';

  if (visible.edges.length === 0) return;

  const maxVar = Math.max(...visible.edges.map(e => e.variance));

  for (const e of visible.edges) {{
    const srcI = document.getElementById('node-' + e.layer_idx + '-' + e.src_i);
    const srcJ = (e.src_i !== e.src_j) ? document.getElementById('node-' + e.layer_idx + '-' + e.src_j) : null;
    const dst = document.getElementById('node-' + (e.layer_idx + 1) + '-' + e.dst_k);
    if (!srcI || !dst) continue;

    const rI = srcI.getBoundingClientRect();
    const rD = dst.getBoundingClientRect();
    const color = e.weight > 0 ? '#e94560' : '#4361ee';
    const relVar = e.variance / maxVar;
    const width = 1 + 4 * relVar;
    const opacity = 0.15 + 0.85 * relVar;
    const dashed = (e.src_i !== e.src_j);

    const dx = rD.left - rect.left + container.scrollLeft;
    const dy = (rD.top + rD.bottom) / 2 - rect.top + container.scrollTop;

    if (!dashed) {{
      const sx = rI.right - rect.left + container.scrollLeft;
      const sy = (rI.top + rI.bottom) / 2 - rect.top + container.scrollTop;
      makePath(svg, sx, sy, dx, dy, color, width, opacity, false, e);
    }} else if (srcJ) {{
      const rJ = srcJ.getBoundingClientRect();
      const sx1 = rI.right - rect.left + container.scrollLeft;
      const sy1 = (rI.top + rI.bottom) / 2 - rect.top + container.scrollTop;
      const sx2 = rJ.right - rect.left + container.scrollLeft;
      const sy2 = (rJ.top + rJ.bottom) / 2 - rect.top + container.scrollTop;
      makePath(svg, sx1, sy1, dx, dy, color, width * 0.7, opacity * 0.8, true, e);
      makePath(svg, sx2, sy2, dx, dy, color, width * 0.7, opacity * 0.8, true, e);
    }}
  }}
}}

function makePath(svg, sx, sy, dx, dy, color, width, opacity, dashed, e) {{
  const gap = dx - sx;
  const cx1 = sx + gap * 0.35;
  const cx2 = dx - gap * 0.35;
  const d = `M${{sx}},${{sy}} C${{cx1}},${{sy}} ${{cx2}},${{dy}} ${{dx}},${{dy}}`;

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', color);
  path.setAttribute('stroke-width', width);
  path.setAttribute('opacity', opacity);
  if (dashed) path.setAttribute('stroke-dasharray', '6,3');
  path.dataset.srcI = e.src_i;
  path.dataset.srcJ = e.src_j;
  path.dataset.dstK = e.dst_k;
  path.dataset.layerIdx = e.layer_idx;
  path.dataset.origOpacity = opacity;
  path.dataset.origWidth = width;
  svg.appendChild(path);
}}

function highlightEdges(bond, feat) {{
  const paths = document.querySelectorAll('#edges path');
  const nodes = document.querySelectorAll('.feature-node');

  const connectedNodes = new Set();
  connectedNodes.add(bond + '-' + feat);

  paths.forEach(p => {{
    const li = parseInt(p.dataset.layerIdx);
    const si = parseInt(p.dataset.srcI);
    const sj = parseInt(p.dataset.srcJ);
    const dk = parseInt(p.dataset.dstK);

    let connected = false;
    if (li === bond && (si === feat || sj === feat)) connected = true;
    if (li === bond - 1 && dk === feat) connected = true;

    if (connected) {{
      p.setAttribute('opacity', Math.min(1, parseFloat(p.dataset.origOpacity) + 0.4));
      p.setAttribute('stroke-width', parseFloat(p.dataset.origWidth) * 1.6);
      connectedNodes.add(li + '-' + si);
      connectedNodes.add(li + '-' + sj);
      connectedNodes.add((li + 1) + '-' + dk);
    }} else {{
      p.setAttribute('opacity', 0.04);
    }}
  }});

  nodes.forEach(n => {{
    const key = n.dataset.bond + '-' + n.dataset.feat;
    if (!connectedNodes.has(key)) {{
      n.classList.add('dimmed');
    }} else {{
      n.classList.add('highlighted');
    }}
  }});
}}

function clearHighlights() {{
  document.querySelectorAll('#edges path').forEach(p => {{
    p.setAttribute('opacity', p.dataset.origOpacity);
    p.setAttribute('stroke-width', p.dataset.origWidth);
  }});
  document.querySelectorAll('.feature-node').forEach(n => {{
    n.classList.remove('dimmed', 'highlighted');
  }});
}}

// ── Click modal ──

function showFeatureModal(bond, feat) {{
  const key = bond + '_' + feat;
  const meta = DATA.features[key];
  const overlay = document.getElementById('modal-overlay');
  const body = document.getElementById('modal-body');

  let html = '';

  const cardPath = 'feature_cards_folded/card_bond' + bond + '_feat' + feat + '.png';
  html += `<img src="${{cardPath}}" onerror="this.style.display='none'" style="max-width:800px;">`;

  html += '<div class="modal-info">';
  const nBonds = DATA.ranks.length;
  if (bond === nBonds - 1) {{
    html += `<h3>Class ${{feat}} (Bond ${{bond}})</h3>`;
  }} else {{
    html += `<h3>Feature ${{feat}} (Bond ${{bond}})</h3>`;
  }}

  if (meta) {{
    html += `<p>Variance: ${{meta.variance.toFixed(4)}}<br>`;
    html += `Mean: ${{meta.mean.toFixed(4)}}<br>`;
    html += `Active: +${{(meta.frac_pos*100).toFixed(1)}}% / -${{(meta.frac_neg*100).toFixed(1)}}%<br>`;
    html += `Mean importance (across classes): ${{(meta.mean_importance || 0).toFixed(4)}}</p>`;

    if (meta.class_selectivity) {{
      const maxSel = Math.max(...meta.class_selectivity.map(Math.abs), 0.01);
      html += '<div class="sel-bar">';
      for (let c = 0; c < 10; c++) {{
        const v = meta.class_selectivity[c];
        const h = Math.round(Math.abs(v) / maxSel * 40);
        const color = CLASS_COLORS[c];
        html += `<div class="sel-bar-item"><div class="sel-bar-fill" style="height:${{h}}px;background:${{color}};margin-top:${{v>=0 ? 40-h : 40}}px"></div><div>${{c}}</div></div>`;
      }}
      html += '</div>';
      html += '<p style="font-size:11px;color:#888;margin-top:4px;">Class selectivity (mean activation per class)</p>';
    }}
  }}

  // Show circuit edges
  const circuit = DATA.circuits[currentClass];
  const visEdges = computeVisibleFeatures(circuit, currentThreshold).edges;
  const incoming = visEdges.filter(e => e.layer_idx === bond - 1 && e.dst_k === feat);
  const outgoing = visEdges.filter(e => e.layer_idx === bond && (e.src_i === feat || e.src_j === feat));

  if (incoming.length > 0) {{
    html += `<p style="margin-top:8px;"><strong>Incoming edges (${{incoming.length}}):</strong><br>`;
    incoming.sort((a,b) => b.variance - a.variance);
    for (const e of incoming.slice(0, 15)) {{
      const sign = e.weight > 0 ? '+' : '';
      const cross = e.src_i !== e.src_j ? ` (f${{e.src_i}} &times; f${{e.src_j}})` : ` (f${{e.src_i}}&sup2;)`;
      html += `w=${{sign}}${{e.weight.toFixed(4)}}, var=${{e.variance.toFixed(4)}}${{cross}}<br>`;
    }}
    if (incoming.length > 15) html += `... and ${{incoming.length - 15}} more`;
    html += '</p>';
  }}

  if (outgoing.length > 0) {{
    html += `<p style="margin-top:8px;"><strong>Outgoing edges (${{outgoing.length}}):</strong><br>`;
    outgoing.sort((a,b) => b.variance - a.variance);
    for (const e of outgoing.slice(0, 15)) {{
      const sign = e.weight > 0 ? '+' : '';
      html += `&rarr; f${{e.dst_k}} (bond ${{bond+1}}): w=${{sign}}${{e.weight.toFixed(4)}}, var=${{e.variance.toFixed(4)}}<br>`;
    }}
    if (outgoing.length > 15) html += `... and ${{outgoing.length - 15}} more`;
    html += '</p>';
  }}

  html += '</div>';
  body.innerHTML = html;
  overlay.classList.add('visible');
}}

function closeModal(e) {{
  if (e.target === document.getElementById('modal-overlay') ||
      e.target.classList.contains('close-btn') ||
      e.key === 'Escape') {{
    document.getElementById('modal-overlay').classList.remove('visible');
  }}
}}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""

    return html


# ================================================================
# Main
# ================================================================

def main():
    t0 = time.time()

    # Step 1: Load model
    print("Step 1: Loading sparse model...")
    ckpt = load_sparse_model(CHECKPOINT)
    cores = ckpt['cores']
    embed = ckpt['embed']
    unembed = ckpt['unembed']
    ranks = ckpt['ranks']
    acc = ckpt.get('acc_finetuned', ckpt.get('acc_pruned', 0)) * 100
    masks = ckpt['masks']
    total_nnz = sum(m.sum().item() for m in masks)
    total_params = sum(m.numel() for m in masks)
    sparsity_pct = 100 * (1 - total_nnz / total_params)

    print(f"  Ranks: {ranks}, acc: {acc:.1f}%, sparsity: {sparsity_pct:.0f}%")
    print(f"  Core shapes: {[c.shape for c in cores]}")

    # Load test data
    print("  Loading SVHN test data...")
    _, test_loader, _ = get_svhn_loaders(data_root=DATA_ROOT)

    # Compute activations
    print("  Computing activations on {0} samples...".format(MAX_SAMPLES))
    activations, labels, images = compute_activations(
        cores, embed, test_loader, DEVICE, MAX_SAMPLES)
    print(f"  Bonds: {[a.shape for a in activations]}")

    # Step 2: Trace circuits per class
    print("\nStep 2: Tracing circuits per class...")
    circuits = []
    for c in range(10):
        circ = trace_class_circuit(cores, activations, c, var_threshold=0.99)
        n_edges = sum(len(layer['edges']) for layer in circ['layers'])
        n_feats = sum(len(v) for v in circ['selected_features'].values())
        print(f"  Class {c}: {n_edges} edges, {n_feats} features")
        circuits.append(circ)

    # Step 2b: Compute cross-class importance for class-specificity ordering
    print("\n  Computing cross-class importance...")
    all_class_imp = {}  # (bond, feat) -> [imp_c0, ..., imp_c9]
    for c, circ in enumerate(circuits):
        imp = compute_importance_for_circuit(circ, threshold=0.99)
        for bond, feat_imps in imp.items():
            for feat, score in feat_imps.items():
                key = (bond, feat)
                if key not in all_class_imp:
                    all_class_imp[key] = [0.0] * 10
                all_class_imp[key][c] = score

    # Print top class-specific features at Bond 2
    print("  Bond 2 class-specificity (imp - mean):")
    for c in range(10):
        b2_feats = []
        for (bond, feat), imps in all_class_imp.items():
            if bond != 2:
                continue
            mean_imp = sum(imps) / 10
            spec = imps[c] - mean_imp
            b2_feats.append((feat, imps[c], mean_imp, spec))
        b2_feats.sort(key=lambda x: -x[3])
        top3 = ', '.join(f'f{f}(spec={s:.2f}, imp={i:.2f})' for f, i, m, s in b2_feats[:3])
        print(f"    Class {c}: {top3}")

    # Step 3: Feature metadata
    print("\nStep 3: Computing feature metadata + thumbnails...")
    all_needed = set()
    for circ in circuits:
        for bond, feats in circ['selected_features'].items():
            for f in feats:
                all_needed.add((bond, f))

    print(f"  {len(all_needed)} unique features across all classes")

    feature_meta = {}
    n_classes = int(labels.max()) + 1
    for idx, (bond, feat) in enumerate(sorted(all_needed)):
        a = activations[bond][:, feat]
        class_sel = []
        for c in range(n_classes):
            mask = labels == c
            class_sel.append(float(a[mask].mean()) if mask.any() else 0.0)

        pattern = trace_feature_to_input(cores, embed, bond, feat)
        thumb = make_thumbnail_b64(pattern)

        # Mean importance across all classes (for class-specificity sorting)
        imps = all_class_imp.get((bond, feat), [0.0] * 10)
        mean_imp = sum(imps) / 10

        feature_meta[(bond, feat)] = {
            'bond': bond,
            'feat': feat,
            'variance': float(np.var(a)),
            'mean': float(np.mean(a)),
            'frac_pos': float((a > 0.4).mean()),
            'frac_neg': float((a < -0.4).mean()),
            'class_selectivity': class_sel,
            'thumbnail_b64': thumb,
            'mean_importance': mean_imp,
        }

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(all_needed)} features processed...")

    print(f"  Done: {len(feature_meta)} features")

    # Step 4: Generate HTML
    print("\nStep 4: Generating HTML...")
    html = generate_html(circuits, feature_meta, ranks, acc, sparsity_pct)

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_HTML) / 1024 / 1024
    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s")
    print(f"Output: {OUTPUT_HTML} ({size_mb:.1f} MB)")
    print(f"Open in browser to view circuits for all 10 classes.")


if __name__ == '__main__':
    main()
