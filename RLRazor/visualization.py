"""
Visualization tools for circuit analysis results.
Creates publication-quality plots comparing SFT vs RL circuit preservation.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Global style ──────────────────────────────────────────────────────────────
# Clean, publication-ready defaults applied once at import time.
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.color':         '#e0e0e0',
    'grid.linewidth':     0.6,
    'grid.alpha':         1.0,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'legend.frameon':     False,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
})

# Consistent color palette across all plots
PAL = {
    'base':  '#2166AC',   # strong blue
    'sft':   '#D6604D',   # muted red-orange
    'rl':    '#1A7741',   # strong green
    'both':  '#F4A736',   # amber  (overlap)
    'gray':  '#AAAAAA',
    'grid':  '#E5E5E5',
    'ref':   '#555555',   # reference line
}


def _save(fig, save_path, extra_msg=''):
    if save_path:
        fig.savefig(save_path)
        print(f"Saved {extra_msg or save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def load_circuit_results(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


# ── 1. Circuit overlap ────────────────────────────────────────────────────────
def plot_circuit_overlap(results, save_path=None):
    """Venn-style bar chart: exclusive and shared head counts."""
    base_heads = set((h['layer'], h['head']) for h in results.get('base_circuit', []))
    sft_heads  = set((h['layer'], h['head']) for h in results.get('sft_circuit', []))
    rl_heads   = set((h['layer'], h['head']) for h in results.get('rl_circuit', []))

    categories = ['Base\nExclusive', 'SFT\nExclusive', 'RL\nExclusive',
                  'SFT ∩ Base', 'RL ∩ Base', 'SFT ∩ RL', 'All Three']
    counts = [
        len(base_heads - sft_heads - rl_heads),
        len(sft_heads  - base_heads - rl_heads),
        len(rl_heads   - base_heads - sft_heads),
        len((sft_heads & base_heads) - rl_heads),
        len((rl_heads  & base_heads) - sft_heads),
        len((sft_heads & rl_heads)   - base_heads),
        len(base_heads & sft_heads & rl_heads),
    ]
    colors = [PAL['base'], PAL['sft'], PAL['rl'],
              '#6BAED6', '#74C476', '#FD8D3C', '#9E9AC8']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.8, width=0.6)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Attention Heads')
    ax.set_title('Circuit Overlap: Base · SFT · RL')
    ax.set_ylim(0, max(counts) * 1.18 if counts else 10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path, 'circuit overlap plot')


# ── 2. CMAP comparison (legacy — kept for backward compat) ───────────────────
def plot_cmap_comparison(results, save_path=None):
    cmap_data = results.get('cmap_analysis', {})
    if not cmap_data or not cmap_data.get('head_info'):
        print("No CMAP data available to plot")
        return

    head_labels = [f"L{h['layer']}H{h['head']}" for h in cmap_data['head_info']]
    sft_deltas  = np.array(cmap_data['sft_deltas'])
    rl_deltas   = np.array(cmap_data['rl_deltas'])
    x           = np.arange(len(head_labels))
    w           = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w/2, sft_deltas, w, label='SFT', color=PAL['sft'])
    ax1.bar(x + w/2, rl_deltas,  w, label='RL',  color=PAL['rl'])
    ax1.axhline(0, color=PAL['ref'], linewidth=0.8, linestyle='--')
    ax1.set_xticks(x); ax1.set_xticklabels(head_labels, rotation=90, fontsize=7)
    ax1.set_ylabel('ΔF'); ax1.set_title('Cross-Model Activation Patching')
    ax1.legend()

    diff   = rl_deltas - sft_deltas
    colors = [PAL['rl'] if d > 0 else PAL['sft'] for d in diff]
    ax2.bar(x, diff, color=colors)
    ax2.axhline(0, color=PAL['ref'], linewidth=0.8, linestyle='--')
    ax2.set_xticks(x); ax2.set_xticklabels(head_labels, rotation=90, fontsize=7)
    ax2.set_ylabel('RL ΔF − SFT ΔF')
    ax2.set_title('Differential Preservation\n(green = RL better)')

    fig.tight_layout()
    _save(fig, save_path, 'CMAP comparison plot')


def plot_cmap_comparison_binary(results, threshold=0.01, save_path=None):
    cmap_data = results.get('cmap_analysis', {})
    if not cmap_data or not cmap_data.get('head_info'):
        print("No CMAP data available to plot")
        return

    head_labels  = [f"L{h['layer']}H{h['head']}" for h in cmap_data['head_info']]
    sft_bin      = (np.abs(np.array(cmap_data['sft_deltas'])) > threshold).astype(int)
    rl_bin       = (np.abs(np.array(cmap_data['rl_deltas']))  > threshold).astype(int)
    x            = np.arange(len(head_labels))
    w            = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w/2, sft_bin, w, label='SFT', color=PAL['sft'])
    ax1.bar(x + w/2, rl_bin,  w, label='RL',  color=PAL['rl'])
    ax1.set_xticks(x); ax1.set_xticklabels(head_labels, rotation=90, fontsize=7)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(['Inactive', 'Active'])
    ax1.set_title(f'Binary Circuit Activation (threshold={threshold})')
    ax1.legend()

    diff   = rl_bin - sft_bin
    colors = [PAL['rl'] if d > 0 else (PAL['sft'] if d < 0 else PAL['gray']) for d in diff]
    ax2.bar(x, diff, color=colors)
    ax2.axhline(0, color=PAL['ref'], linewidth=0.8, linestyle='--')
    ax2.set_xticks(x); ax2.set_xticklabels(head_labels, rotation=90, fontsize=7)
    ax2.set_yticks([-1, 0, 1]); ax2.set_yticklabels(['SFT only', 'Same', 'RL only'])
    ax2.set_title('Binary Differential Preservation')

    fig.tight_layout()
    _save(fig, save_path, 'binary CMAP comparison plot')


# ── 3. Vulnerable circuits ────────────────────────────────────────────────────
def plot_vulnerable_circuits(results, save_path=None):
    vulnerable = results.get('vulnerable_circuits', [])
    fig, ax = plt.subplots(figsize=(10, 5))
    if not vulnerable:
        print("No vulnerable circuits found!")
        ax.text(0.5, 0.5, 'No vulnerable circuits identified',
                ha='center', va='center', fontsize=13, color=PAL['gray'])
        ax.set_axis_off()
        _save(fig, save_path)
        return

    top     = vulnerable[:15]
    labels  = [f"L{h['layer']}H{h['head']}" for h in top]
    x, w    = np.arange(len(labels)), 0.25
    ax.bar(x - w, [h['sft_delta'] for h in top], w, label='SFT ΔF', color=PAL['sft'])
    ax.bar(x,     [h['rl_delta']  for h in top], w, label='RL ΔF',  color=PAL['rl'])
    ax.bar(x + w, [h['vulnerability'] for h in top], w,
           label='Vulnerability', color='#B2182B')
    ax.axhline(0, color=PAL['ref'], linewidth=0.8, linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Score'); ax.set_title('Most Vulnerable Circuits')
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path, 'vulnerable circuits plot')


# ── 4. Circuit heatmap ────────────────────────────────────────────────────────
def plot_circuit_heatmap(results, model_type='base', save_path=None):
    circuit = results.get(f'{model_type}_circuit', [])
    if not circuit:
        print(f"No circuit data for {model_type}")
        return

    max_layer = max(h['layer'] for h in circuit)
    max_head  = max(h['head']  for h in circuit)
    mat       = np.zeros((max_layer + 1, max_head + 1))
    for h in circuit:
        mat[h['layer'], h['head']] = h.get('mask_value', abs(h.get('importance_score', 0.0)))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mat, cmap='Blues', linewidths=0,
                cbar_kws={'label': 'DBM Mask Value', 'shrink': 0.8},
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer')
    ax.set_title(f'{model_type.upper()} — Circuit Heatmap (DBM Mask Values)')
    fig.tight_layout()
    _save(fig, save_path, f'{model_type} heatmap')


# ── 5. Faithfulness comparison ────────────────────────────────────────────────
def plot_faithfulness_comparison(results, save_path=None):
    faith = results.get('faithfulness', {})
    if not faith:
        print("No faithfulness data available")
        return

    models  = list(faith.keys())
    f_scores = [faith[m].get('faithfulness', 0)      for m in models]
    fracs    = [faith[m].get('circuit_fraction', 0) * 100 for m in models]
    colors   = [PAL.get(m, PAL['gray']) for m in models]
    x        = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Faithfulness score
    bars = ax1.bar(x, f_scores, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, f_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels([m.upper() for m in models])
    ax1.set_ylabel('Faithfulness  F(C|M) / F(M)')
    ax1.set_title('Circuit Faithfulness')
    ax1.set_ylim(0, max(f_scores) * 1.2 if f_scores else 1.5)
    ax1.axhline(1.0, color=PAL['ref'], linewidth=0.8, linestyle='--', label='F = 1 (perfect)')
    ax1.legend()

    # Circuit efficiency (size)
    bars2 = ax2.bar(x, fracs, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars2, fracs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels([m.upper() for m in models])
    ax2.set_ylabel('Circuit Size (% of all heads)')
    ax2.set_title('Circuit Efficiency\n(smaller = more concentrated)')
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    _save(fig, save_path, 'faithfulness comparison plot')


# ── 6. DCM analysis ───────────────────────────────────────────────────────────
def plot_dcm_analysis(results, save_path=None):
    dcm_data = results.get('dcm_analysis', {})
    if not dcm_data:
        print("No DCM data available")
        return

    models     = list(dcm_data.keys())
    first      = models[0]
    hypotheses = list(dcm_data[first].keys()) if dcm_data[first] else []
    if not hypotheses:
        print("No hypotheses in DCM results")
        return

    n_hyp    = len(hypotheses)
    colors   = [PAL.get(m, PAL['gray']) for m in models]
    x        = np.arange(len(models))

    fig, axes = plt.subplots(1, n_hyp, figsize=(4.5 * n_hyp, 5), sharey=True)
    if n_hyp == 1:
        axes = [axes]

    for ax, hyp in zip(axes, hypotheses):
        counts = []
        for m in models:
            res = dcm_data.get(m, {}).get(hyp, {})
            counts.append(len(res.get('active_heads', [])) if isinstance(res, dict) else 0)

        bars = ax.bar(x, counts, color=colors, edgecolor='white', width=0.5)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(hyp.replace('_', ' ').title())
        if ax is axes[0]:
            ax.set_ylabel('Active Heads')

    fig.suptitle('DCM Analysis — Heads Encoding Each Functionality', y=1.02)
    fig.tight_layout()
    _save(fig, save_path, 'DCM analysis plot')


# ── 7. Binary analysis summary ────────────────────────────────────────────────
def plot_binary_analysis(results, save_path=None):
    bd = results.get('binary_analysis', {})
    if not bd:
        print("No binary analysis data available")
        return

    config  = results.get('config', {})
    label_a = config.get('model_a_name', 'sft')
    label_b = config.get('model_b_name', 'rl')

    models   = [label_a.upper(), label_b.upper()]
    overlaps = [bd.get(f'{label_a}_overlap_pct', 0), bd.get(f'{label_b}_overlap_pct', 0)]
    counts   = [bd.get(f'{label_a}_overlap_count', 0), bd.get(f'{label_b}_overlap_count', 0)]
    base_sz  = bd.get('base_circuit_size', 0)
    colors   = [PAL['sft'], PAL['rl']]
    x        = np.arange(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # % overlap
    bars = ax1.bar(x, overlaps, color=colors, edgecolor='white', width=0.45)
    for bar, v in zip(bars, overlaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Overlap with Base Circuit (%)')
    ax1.set_title('Base Circuit Preservation')

    adv = bd.get(f'{label_b}_advantage', 0)
    if adv > 0:
        ax1.annotate(f'+{adv:.1f} pp\nadvantage',
                     xy=(1, overlaps[1]), xytext=(1.35, overlaps[1] - 8),
                     fontsize=9, color=PAL['rl'],
                     arrowprops=dict(arrowstyle='->', color=PAL['rl'], lw=1.2))

    # absolute counts
    ax2.bar(x - 0.2, [base_sz, base_sz], 0.35, label='Base circuit', color=PAL['base'], alpha=0.35)
    ax2.bar(x + 0.2, counts, 0.35, label='Shared with base', color=colors, alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.set_ylabel('Number of Heads')
    ax2.set_title('Head Counts')
    ax2.legend()

    fig.tight_layout()
    _save(fig, save_path, 'binary analysis plot')


# ── 8. Per-head circuit presence (diverging) ──────────────────────────────────
def plot_binary_differential(results, save_path=None):
    """
    Diverging bar per head: RL active = green bar UP, SFT active = red bar DOWN.
    Overlap (both active) = full candle.  Neither = empty gap.
    All 576 heads plotted so inactive heads appear as visible empty space.
    """
    masks = results.get('binary_masks')

    def _parse(mask_dict):
        out = {}
        for k, v in mask_dict.items():
            s = str(k).strip()
            if s.startswith('(') and ',' in s:
                try:
                    layer = int(s[s.find('(')+1:s.find(',')])
                    head  = int(s[s.find(',')+1:s.find(')')])
                    out[(layer, head)] = int(v); continue
                except Exception:
                    pass
            if '-' in s:
                p = s.split('-')
                if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
                    out[(int(p[0]), int(p[1]))] = int(v); continue
        return out

    if isinstance(masks, dict) and all(k in masks for k in ['base', 'sft', 'rl']):
        sft_mask = _parse(masks['sft'])
        rl_mask  = _parse(masks['rl'])
        all_heads = set(sft_mask) | set(rl_mask)
    else:
        sft_heads = set((h['layer'], h['head']) for h in results.get('sft_circuit', []))
        rl_heads  = set((h['layer'], h['head']) for h in results.get('rl_circuit', []))
        all_heads = sft_heads | rl_heads
        sft_mask  = {h: int(h in sft_heads) for h in all_heads}
        rl_mask   = {h: int(h in rl_heads)  for h in all_heads}

    # All possible heads in model
    max_layer = max(h[0] for h in all_heads) + 1 if all_heads else 36
    max_head  = max(h[1] for h in all_heads) + 1 if all_heads else 16
    all_possible = [(l, h) for l in range(max_layer) for h in range(max_head)]

    head_labels = [f"L{h[0]}H{h[1]}" for h in all_possible]
    rl_up  = [ rl_mask.get(h, 0)  for h in all_possible]
    sft_dn = [-sft_mask.get(h, 0) for h in all_possible]

    n_heads   = len(head_labels)
    fig_width = max(24, n_heads * 0.18)

    # 8a — diverging per-head chart
    fig1, ax1 = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(n_heads)

    ax1.bar(x, rl_up,  color=PAL['rl'],  alpha=0.9, label='RL active (↑)')
    ax1.bar(x, sft_dn, color=PAL['sft'], alpha=0.9, label='SFT active (↓)')
    ax1.axhline(0, color=PAL['ref'], linewidth=0.8)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['SFT active', '', 'RL active'], fontsize=10)
    ax1.set_xticks(x)
    tick_fs = max(4, min(8, int(260 / n_heads)))
    ax1.set_xticklabels(head_labels, rotation=90, ha='right', fontsize=tick_fs)
    ax1.set_xlabel('Attention Head (Layer × Head)')
    ax1.set_title(
        'Circuit Presence per Head: RL (green ↑) vs SFT (red ↓)\n'
        'Both bars = shared head  |  One bar = exclusive  |  Empty = neither active'
    )
    ax1.legend(loc='upper right')
    # suppress y-axis grid, keep x clean
    ax1.yaxis.grid(True, color=PAL['grid']); ax1.set_axisbelow(True)
    fig1.tight_layout()

    if save_path:
        stem = save_path.rsplit('.', 1)[0]
        fig1.savefig(f"{stem}_perhead.png")
        print(f"Saved per-head plot to {stem}_perhead.png")
        plt.close(fig1)
        dist_path = f"{stem}_distribution.png"
    else:
        plt.show(); plt.close(fig1)
        dist_path = None

    # 8b — summary 4-category bar
    n_both     = sum(1 for h in all_possible if sft_mask.get(h,0) and rl_mask.get(h,0))
    n_sft_only = sum(1 for h in all_possible if sft_mask.get(h,0) and not rl_mask.get(h,0))
    n_rl_only  = sum(1 for h in all_possible if not sft_mask.get(h,0) and rl_mask.get(h,0))
    n_neither  = sum(1 for h in all_possible if not sft_mask.get(h,0) and not rl_mask.get(h,0))

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    cat_vals   = [n_sft_only, n_rl_only, n_both, n_neither]
    cat_labels = ['SFT\nExclusive', 'RL\nExclusive', 'Shared\n(overlap)', 'Neither\nActive']
    cat_colors = [PAL['sft'], PAL['rl'], PAL['both'], PAL['gray']]

    bars2 = ax2.bar(cat_labels, cat_vals, color=cat_colors, edgecolor='white', width=0.55)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2,
                 str(h), ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Number of Attention Heads')
    ax2.set_title('Circuit Membership Summary: SFT vs RL')
    ax2.yaxis.grid(True, color=PAL['grid']); ax2.set_axisbelow(True)
    fig2.tight_layout()

    if dist_path:
        fig2.savefig(dist_path)
        print(f"Saved distribution plot to {dist_path}")
        plt.close(fig2)
    else:
        plt.show(); plt.close(fig2)


# ── 9. DBM mask comparison ────────────────────────────────────────────────────
def plot_circuit_overlap_dbm(results, save_path=None):
    """Grouped bar: DBM mask values for every base-circuit head across models."""
    config  = results.get('config', {})
    label_a = config.get('model_a_name', 'sft')
    label_b = config.get('model_b_name', 'rl')

    base_circuit = results.get('base_circuit', [])
    if not base_circuit:
        print("No base circuit — skipping DBM mask comparison plot")
        return

    base_sorted = sorted(base_circuit, key=lambda h: h.get('mask_value', 0), reverse=True)
    head_labels = [f"L{h['layer']}H{h['head']}" for h in base_sorted]
    n = len(head_labels)

    def lkp(key):
        return {(h['layer'], h['head']): h.get('mask_value', 0.0)
                for h in results.get(key, [])}

    a_lkp = lkp(f'{label_a}_circuit')
    b_lkp = lkp(f'{label_b}_circuit')

    base_m = [h.get('mask_value', 0.0) for h in base_sorted]
    a_m    = [a_lkp.get((h['layer'], h['head']), 0.0) for h in base_sorted]
    b_m    = [b_lkp.get((h['layer'], h['head']), 0.0) for h in base_sorted]

    x, w = np.arange(n), 0.26
    fig, ax = plt.subplots(figsize=(max(12, n * 0.65), 5))

    ax.bar(x - w, base_m, w, label='Base',           color=PAL['base'], alpha=0.9)
    ax.bar(x,     a_m,    w, label=label_a.upper(),   color=PAL['sft'],  alpha=0.9)
    ax.bar(x + w, b_m,    w, label=label_b.upper(),   color=PAL['rl'],   alpha=0.9)

    ax.axhline(0.5, color=PAL['ref'], linewidth=0.9, linestyle='--', label='Active threshold (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=45, ha='right',
                       fontsize=max(5, min(9, int(240 / n))))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('DBM Mask Value')
    ax.set_title('DBM Mask Values per Base-Circuit Head\n'
                 f'Base · {label_a.upper()} · {label_b.upper()}  —  heads sorted by base mask (desc.)')
    ax.legend(loc='upper right')
    ax.yaxis.grid(True, color=PAL['grid']); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path, 'DBM mask comparison plot')


# ── 10. Head contribution graph ───────────────────────────────────────────────
def plot_head_contribution_graph(results, save_path=None):
    """
    PI-requested contribution graph.
    One subplot per model (base / SFT / RL).
    Bar = DBM mask value, ▲ = necessity (normalised), ● = sufficiency (normalised).
    """
    config   = results.get('config', {})
    label_a  = config.get('model_a_name', 'sft')
    label_b  = config.get('model_b_name', 'rl')

    base_circuit = results.get('base_circuit', [])
    if not base_circuit:
        print("No base circuit data — skipping contribution graph")
        return

    base_sorted = sorted(base_circuit, key=lambda h: h.get('mask_value', 0), reverse=True)
    head_labels = [f"L{h['layer']}H{h['head']}" for h in base_sorted]
    n = len(head_labels)
    x = np.arange(n)

    def mask_lkp(key):
        return {(h['layer'], h['head']): h.get('mask_value', 0.0)
                for h in results.get(key, [])}

    def ns_lkp(model_key):
        per = results.get('necessity_sufficiency', {}).get(model_key, {}).get('per_head', {})
        return {(v['layer'], v['head']): v for v in per.values()}

    a_lkp  = mask_lkp(f'{label_a}_circuit')
    b_lkp  = mask_lkp(f'{label_b}_circuit')
    base_ns = ns_lkp('base')
    a_ns    = ns_lkp(label_a)
    b_ns    = ns_lkp(label_b)

    def get_ns(d, h, field):
        return d.get((h['layer'], h['head']), {}).get(field, None)

    rows = [
        ('Base',           PAL['base'], [h.get('mask_value', 0.0) for h in base_sorted], base_ns),
        (label_a.upper(),  PAL['sft'],  [a_lkp.get((h['layer'], h['head']), 0.0) for h in base_sorted], a_ns),
        (label_b.upper(),  PAL['rl'],   [b_lkp.get((h['layer'], h['head']), 0.0) for h in base_sorted], b_ns),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.75), 13), sharex=True)
    fig.suptitle('Head Contribution: Mask Value · Necessity · Sufficiency\n'
                 '(heads sorted by base mask value, descending)', fontsize=12)

    for ax, (label, color, masks, ns_dict) in zip(axes, rows):
        ax.bar(x, masks, color=color, alpha=0.75, width=0.6, label='Mask value')
        ax.axhline(0.5, color=PAL['ref'], linewidth=0.8, linestyle='--', alpha=0.6)

        nec = [get_ns(ns_dict, h, 'necessity') for h in base_sorted]
        vx  = [xi for xi, v in zip(x, nec) if v is not None]
        vn  = [v  for v in nec if v is not None]
        if vn:
            lo, hi = min(vn), max(vn)
            norm = [(v - lo) / (hi - lo + 1e-10) for v in vn]
            ax.scatter(vx, norm, marker='^', color='#B2182B', s=55, zorder=5,
                       label='Necessity (norm.)', linewidths=0)

        suf = [get_ns(ns_dict, h, 'sufficiency_logprob') for h in base_sorted]
        vx2 = [xi for xi, v in zip(x, suf) if v is not None]
        vs  = [v  for v in suf if v is not None]
        if vs:
            lo2, hi2 = min(vs), max(vs)
            norm2 = [(v - lo2) / (hi2 - lo2 + 1e-10) for v in vs]
            ax.scatter(vx2, norm2, marker='o', color='#2CA25F', s=35, zorder=5,
                       label='Sufficiency (norm.)', linewidths=0)

        ax.set_ylabel('Value')
        ax.set_ylim(-0.05, 1.2)
        ax.set_title(label, fontsize=11, color=color, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=3)
        ax.yaxis.grid(True, color=PAL['grid']); ax.set_axisbelow(True)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(head_labels, rotation=45, ha='right',
                              fontsize=max(5, min(9, int(220 / n))))
    axes[-1].set_xlabel('Attention Head (sorted by base mask value)')
    fig.tight_layout()
    _save(fig, save_path, 'head contribution graph')


# ── Generate all ──────────────────────────────────────────────────────────────
def generate_all_visualizations(results_path: str, output_dir: str = "results/circuits/plots"):
    print(f"Loading results from {results_path}...")
    results = load_circuit_results(results_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    task = results.get('config', {}).get('task', 'unknown')

    print("\nGenerating visualizations...")

    print("  1. Circuit overlap...")
    plot_circuit_overlap(results, save_path=f"{output_dir}/circuit_overlap_{task}.png")

    print("  2a. CMAP comparison...")
    plot_cmap_comparison(results, save_path=f"{output_dir}/cmap_comparison_{task}.png")

    print("  2b. CMAP binary...")
    plot_cmap_comparison_binary(results, threshold=0.01,
                                save_path=f"{output_dir}/cmap_binary_{task}.png")

    print("  3. Vulnerable circuits...")
    plot_vulnerable_circuits(results, save_path=f"{output_dir}/vulnerable_circuits_{task}.png")

    print("  4. Circuit heatmaps...")
    for m in ['base', 'sft', 'rl']:
        plot_circuit_heatmap(results, model_type=m,
                             save_path=f"{output_dir}/circuit_heatmap_{m}_{task}.png")

    print("  5. Faithfulness comparison...")
    plot_faithfulness_comparison(results, save_path=f"{output_dir}/faithfulness_comparison_{task}.png")

    print("  6. DCM analysis...")
    plot_dcm_analysis(results, save_path=f"{output_dir}/dcm_analysis_{task}.png")

    print("  7. Binary analysis summary...")
    plot_binary_analysis(results, save_path=f"{output_dir}/binary_analysis_{task}.png")

    print("  8. Per-head circuit presence (diverging)...")
    plot_binary_differential(results, save_path=f"{output_dir}/binary_differential_{task}.png")

    print("  9. DBM mask comparison...")
    plot_circuit_overlap_dbm(results, save_path=f"{output_dir}/mask_comparison_{task}.png")

    print("  10. Head contribution graph...")
    plot_head_contribution_graph(results, save_path=f"{output_dir}/head_contribution_{task}.png")

    print(f"\nAll visualizations saved to {output_dir}/")


# ── Text summary ──────────────────────────────────────────────────────────────
def print_circuit_summary(results_path: str):
    results = load_circuit_results(results_path)
    config  = results.get('config', {})
    label_a = config.get('model_a_name', 'sft')
    label_b = config.get('model_b_name', 'rl')

    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Task:            {config.get('task', 'unknown')}")
    print(f"  Circuit method:  {config.get('circuit_method', 'unknown')}")
    print(f"  Lambda sparsity: {config.get('lambda_sparsity', 'N/A')}")
    print(f"  Max examples:    {config.get('max_examples', 'N/A')}")

    base_h = set((h['layer'], h['head']) for h in results.get('base_circuit', []))
    a_h    = set((h['layer'], h['head']) for h in results.get(f'{label_a}_circuit', []))
    b_h    = set((h['layer'], h['head']) for h in results.get(f'{label_b}_circuit', []))

    print(f"\nCircuit Sizes (DBM mask > 0.5):")
    print(f"  Base: {len(base_h)}  {label_a.upper()}: {len(a_h)}  {label_b.upper()}: {len(b_h)}")

    if base_h:
        ao = len(base_h & a_h); bo = len(base_h & b_h)
        print(f"\nCircuit Overlap with Base:")
        print(f"  {label_a.upper()}: {ao}/{len(base_h)} ({100*ao/len(base_h):.1f}%)")
        print(f"  {label_b.upper()}: {bo}/{len(base_h)} ({100*bo/len(base_h):.1f}%)")

    faith = results.get('faithfulness', {})
    if faith:
        print(f"\nFaithfulness (Equation 4):")
        for m, v in faith.items():
            if isinstance(v, dict):
                print(f"  {m.upper()}: {v.get('faithfulness', 0):.4f}")

    dcm = results.get('dcm_analysis', {})
    if dcm:
        print(f"\nDCM Analysis (Equation 3):")
        for m, hyps in dcm.items():
            if hyps:
                print(f"  {m.upper()}:")
                for hyp, res in hyps.items():
                    if isinstance(res, dict):
                        print(f"    {hyp}: {len(res.get('active_heads', []))} active heads")

    ns_data = results.get('necessity_sufficiency', {})
    if ns_data:
        print(f"\nNecessity & Sufficiency:")
        for m, ns in ns_data.items():
            if isinstance(ns, dict) and 'circuit_necessity' in ns:
                print(f"  {m.upper()}: necessity={ns['circuit_necessity']:.4f}  "
                      f"sufficiency={ns['circuit_sufficiency']:.4f}  ({ns['circuit_size']} heads)")
                top3 = sorted(ns.get('per_head', {}).values(),
                              key=lambda x: x['necessity'], reverse=True)[:3]
                for h in top3:
                    print(f"    L{h['layer']}H{h['head']}: "
                          f"necessity={h['necessity']:.4f}  "
                          f"sufficiency_lp={h['sufficiency_logprob']:.4f}")

    if base_h:
        ao = len(base_h & a_h); bo = len(base_h & b_h)
        winner = label_b.upper() if bo > ao else (label_a.upper() if ao > bo else None)
        diff   = abs(bo - ao)
        print("\n" + "="*70)
        if winner:
            loser = label_a.upper() if winner == label_b.upper() else label_b.upper()
            print(f"KEY FINDING: {winner} preserves base circuits better than {loser}")
            print(f"  {winner} maintains {diff} more base circuit heads")
        else:
            print("KEY FINDING: Models show equivalent circuit preservation")
        print("="*70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_path> [output_dir]")
        sys.exit(1)
    results_path = sys.argv[1]
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "results/circuits/plots"
    print_circuit_summary(results_path)
    generate_all_visualizations(results_path, output_dir)
