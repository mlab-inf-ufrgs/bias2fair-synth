import warnings
warnings.filterwarnings('ignore')

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from utils.data_loader import DATASET_CONFIGS

# ── Estilo científico (IEEE / Springer) ─────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#333333',
    'axes.linewidth':    0.8,
    'axes.labelcolor':   '#222222',
    'axes.labelsize':    10,
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'xtick.major.size':  0,
    'ytick.major.size':  0,
    'text.color':        '#111111',
    'font.family':       'DejaVu Sans',
    'font.size':         10,
})

# Colormap científico: sequencial azul-cinza (sem saturação excessiva)
# 'Blues'  → clássico; 'GnBu' → verde-azul usado em mapas de calor científicos
CMAP = 'Blues'

# Cor da borda da classe favorável: azul escuro discreto (não verde brilhante)
FAV_EDGE  = '#1a3a5c'
FAV_LW    = 2.0
FAV_STYLE = '--'   # tracejado para não confundir com borda do eixo

# ── Rótulos específicos por dataset ──────────────────────────────────────────
DATASET_LABELS = {
    'compas': {
        'target_labels':    ['No Recidivism\n(0)', 'Recidivism\n(1)'],
        'sensitive_labels': ['African-Am.\n(0)', 'Caucasian\n(1)'],
        'xlabel': 'Target: two_year_recid',
        'ylabel': 'Protected attribute: race',
        'favorable_class': 0,
        'disadvantaged':   0,   # linha 0 = grupo desprivilegiado
    },
    'adult': {
        'target_labels':    ['Income \u226450K\n(0)', 'Income >50K\n(1)'],
        'sensitive_labels': ['Female\n(0)', 'Male\n(1)'],
        'xlabel': 'Target: income',
        'ylabel': 'Protected attribute: sex',
        'favorable_class': 1,
        'disadvantaged':   0,
    },
    'diabetes': {
        'target_labels':    ['No Readmission\n(0)', 'Readmission\n(1)'],
        'sensitive_labels': ['African-Am.\n(0)', 'Other\n(1)'],
        'xlabel': 'Target: readmitted',
        'ylabel': 'Protected attribute: race',
        'favorable_class': 0,
        'disadvantaged':   0,
    },
}


def plot_crosstab(ax, matrix, N, lbl):
    """Renderiza a matriz de contingência 2x2 em `ax`."""

    # ── Heatmap ──
    cmap_obj = plt.cm.get_cmap(CMAP)
    # Normalizar para [0.08, 0.85] para evitar branco puro e azul saturado
    norm = mcolors.Normalize(vmin=matrix.min() * 0.5, vmax=matrix.max() * 1.1)
    im   = ax.imshow(matrix, cmap=cmap_obj, norm=norm, aspect='auto')

    # ── Anotações (valor absoluto + %) ──
    for i in range(2):
        for j in range(2):
            val      = matrix[i, j]
            bg_color = cmap_obj(norm(val))
            # Luminância perceptual para decidir cor do texto
            lum = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
            txt_color = 'white' if lum < 0.45 else '#111111'
            ax.text(
                j, i,
                f'{val:,}\n({val / N * 100:.1f}%)',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color=txt_color,
                linespacing=1.4,
            )

    # ── Borda tracejada na coluna favorável ──
    fav = lbl['favorable_class']
    for row in range(2):
        rect = Rectangle(
            (fav - 0.5, row - 0.5), 1, 1,
            linewidth=FAV_LW, edgecolor=FAV_EDGE,
            linestyle=FAV_STYLE, facecolor='none',
        )
        ax.add_patch(rect)

    # ── Eixos ──
    ax.set_xticks([0, 1])
    ax.set_xticklabels(lbl['target_labels'], fontsize=9, color='#222222')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(lbl['sensitive_labels'], fontsize=9, color='#222222')
    ax.set_xlabel(lbl['xlabel'], fontsize=10, labelpad=6, color='#333333')
    ax.set_ylabel(lbl['ylabel'], fontsize=10, labelpad=6, color='#333333')

    # Remover spines desnecessárias
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#555555')

    return im


def main():
    datasets = ['compas', 'adult', 'diabetes']

    out_dir = os.path.join(PROJECT_ROOT, 'plots', 'individual_crosstabs')
    os.makedirs(out_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"Processando {dataset_name.upper()}...")
        cfg  = DATASET_CONFIGS[dataset_name]
        lbl  = DATASET_LABELS[dataset_name]
        data = cfg['loader']()

        y = data[cfg['target']].values
        s = data[cfg['sensitive']].values
        N = len(data)

        cross  = pd.crosstab(s, y)
        matrix = cross.values   # shape (2, 2): rows = sensitive, cols = target

        fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor='white')
        im = plot_crosstab(ax, matrix, N, lbl)

        # Colorbar discreta e compacta
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, length=2)
        cbar.outline.set_linewidth(0.5)

        plt.tight_layout(pad=0.8)

        out_png = os.path.join(out_dir, f'crosstab_{dataset_name}.png')
        out_eps = os.path.join(out_dir, f'crosstab_{dataset_name}.eps')
        fig.savefig(out_png, dpi=250, bbox_inches='tight', facecolor='white')
        fig.savefig(out_eps, format='eps', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  -> {out_png}")

    print(f"\nProntos em: {out_dir}/")


if __name__ == "__main__":
    main()
