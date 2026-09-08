"""Small-multiples map of crust mineralogy over the two composition axes.

One pie per (mantle Mg/Si, core-formation dIW) cell of crust_compositions.csv, showing the
CIPW-normative assemblage the weathering model actually consumes. Reads the table through
kamino.crust_composition so the figure can never disagree with the model.

    python experiments/plot_crust_grid.py [--out output/crust_grid]

Design notes
------------
Pie charts are a weak form for comparing values, and they are used here only because the
question is "how does the assemblage change across the grid", where each panel is read as a
shape rather than as eight numbers. Three things carry identity so it is never colour alone:

  * slice order is IDENTICAL in every panel, so angular position encodes mineral;
  * a legend is always present, and the Earth cell is direct-labelled and ringed;
  * mineral weight fractions AND melt SiO2 are written to a companion CSV as the table view.

Colour encodes the mineral FAMILY and texture the cation within it, so eleven phases are read as
six groups; see COLORS for the mapping and its validation. Slices of one family are adjacent in
MINERALS, so every shared-hue boundary is a texture edge rather than two indistinguishable
wedges, and the slot order still runs silica-rich to silica-poor.

The palette clears the CVD floor under protanopia, deuteranopia AND tritanopia on ALL pairs, not
merely the ring-adjacent ones -- worst 11.8, tritan 9.5, normal-vision 18.8. Hatching is drawn in
the edge colour, so `hatch_ink` flips it to the surface colour on the dark fills; a near-black
texture on the dark blue Fe-pyroxenes is invisible. Do not reassign hues or reorder the minerals
without re-checking, and check all pairs rather than only the adjacent ones: Quartz and the
Fe-pyroxenes dominate the same cells at low Mg/Si without ever touching.
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from kamino.constants import EARTH_DELTA_IW, EARTH_MANTLE_MG_SI
from kamino.crust_composition import CRUST_TABLE, cipw_norm

# Fixed slice order, silica-rich -> silica-poor, with the albite->nepheline cascade pair
# adjacent. Changing this changes which colour pairs touch; see the module docstring.
# Hue encodes the MINERAL FAMILY and texture the cation within it, so the ring is read in two
# steps rather than eleven. Dotted means CALCIUM-BEARING everywhere it appears:
#
#   quartz                    amber
#   feldspar / feldspathoid   green       Albite | Anorthite (Ca, dotted) | Nepheline (lines)
#   pyroxene, Mg              light blue  Enstatite | Diopside (Ca, dotted)
#   pyroxene, Fe              dark blue   Ferrosilite | Hedenbergite (Ca, dotted)
#   melilite                  purple      Akermanite (Ca, dotted)
#   olivine                   red         Fayalite | Forsterite (Mg, lines)
#
# Nepheline is not a plagioclase; it takes the feldspar hue because it IS the desilicated albite
# (norm step 5b converts one into the other), so the cascade reads as a texture change.
#
# Six hues, chosen by exhausting the candidates against the CVD checks on ALL pairs rather than
# only ring-adjacent ones -- Quartz and the Fe-pyroxenes co-occur prominently at low Mg/Si without
# touching, so adjacency alone is too weak a test here. Machado et al. (2009) at severity 1.0,
# OKLab dE x100: worst pair 11.8 (green/red), worst tritan 9.5, worst normal-vision 18.8. The
# green and red are deepened and lightened respectively from the documented ring for exactly that
# reason; at the textbook values the green/red pair sits at 7.2, inside the 6-8 floor band.
MINERALS = ['Quartz', 'Albite', 'Anorthite', 'Nepheline',
            'Enstatite', 'Diopside', 'Ferrosilite', 'Hedenbergite',
            'Akermanite', 'Forsterite', 'Fayalite']
COLORS = {'Quartz': '#eda100',
          'Albite': '#0a5c28', 'Anorthite': '#0a5c28', 'Nepheline': '#0a5c28',
          'Enstatite': '#7fb9e8', 'Diopside': '#7fb9e8',
          'Ferrosilite': '#0d2f6b', 'Hedenbergite': '#0d2f6b',
          'Akermanite': '#8c5a9e',
          'Forsterite': '#ef5350', 'Fayalite': '#ef5350'}
HATCHED = {'Anorthite': '...', 'Nepheline': '///', 'Diopside': '...',
           'Hedenbergite': '...', 'Akermanite': '...', 'Forsterite': '///'}


def hatch_ink(fill):
    """Hatch colour for a fill: light on dark, dark on light, or the texture is invisible.

    Matplotlib draws hatching in the patch's EDGE colour, so a near-black hatch on the dark blue
    Fe-pyroxenes or the dark green feldspars would simply not read.
    """
    srgb = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    return SURFACE if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.22 else INK

INK, INK_2, INK_3 = '#1a1a19', '#55554e', '#8a8a80'
SURFACE, MUTED = '#ffffff', '#e8e8e2'

# Subsampled so the panels stay legible; every value is an exact grid node (no interpolation).
MGSI_SHOW = [0.5, 0.7, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0]
# 0.1 steps from -1.5 up: the quartz-out transition lives there and a 1.0-spaced axis aliases it.
DIW_SHOW = [-5.0, -4.0, -3.0, -2.0, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0]

# The same page style as plot_results.py -- serif type, MNRAS sizes, 300 dpi -- read from the
# style file rather than by importing that module, which runs plt.style.use() at import and
# would also drag in the sweep-loading machinery. constrained_layout is then switched back off:
# the style turns it on, and it silently disables the subplots_adjust this layout depends on.
STYLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'planetary-chem-paper.mplstyle')
plt.style.use(STYLE_FILE)
plt.rcParams['figure.constrained_layout.use'] = False

# MNRAS text width, 504 pt, matching figure_size('double') in plot_results.
TEXT_WIDTH_IN = 504 / 72.27

# --paper: the same figure at MNRAS text width. The transition rows are kept because they are the
# result worth showing; the interior Mg/Si columns go, since they interpolate visibly between the
# ones that remain.
PAPER_MGSI = [0.5, 0.9, 1.25, 1.6, 2.0]
PAPER_DIW = [-5.0, -3.0, -2.0, -1.3, -1.0]


def assemblage(row):
    """(weight-fraction dict, is_mass_violating) for one table row."""
    ox = {'SiO2': row.SiO2, 'TiO2': row.TiO2, 'Al2O3': row.Al2O3, 'FeOt': row.FeO,
          'MgO': row.MgO, 'CaO': row.CaO, 'Na2O': row.Na2O, 'K2O': row.K2O}
    # cipw_norm RAISES when the phase set cannot express the melt (a residual silica deficit
    # after both desilication cascades). That is a real exclusion, not a warning, so catch it
    # and mark the cell rather than letting it kill the figure.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return cipw_norm(ox, emit_quartz=True), False
    except ValueError:
        return {}, True


def draw_grid_axes(fig, axes, mgsi_show, diw_show, fs):
    """Bottom (Mg/Si) and left ($\\Delta$IW) axis lines, one tick per grid column and row.

    The grid is a matrix of panels, so the two composition variables are categorical axes: ticks
    sit at the panel centres, not at value-proportional positions. Drawn in figure coordinates
    after subplots_adjust, since no single axes spans the grid.
    """
    w_in, h_in = fig.get_size_inches()
    inv = fig.transFigure.inverted()
    centre = lambda ax: inv.transform(ax.transData.transform((0, 0)))
    xs = [centre(axes[0][ix])[0] for ix in range(len(mgsi_show))]
    ys = [centre(axes[iy][0])[1] for iy in range(len(diw_show))]
    boxes = [ax.get_position() for row in axes for ax in row]
    x_lo, y_lo = min(b.x0 for b in boxes), min(b.y0 for b in boxes)

    # Everything below is set in inches and converted, so the two figure sizes get the same
    # physical gaps rather than the same fractions.
    def fx(inches):
        return inches / w_in

    def fy(inches):
        return inches / h_in

    def line(x, y):
        fig.add_artist(Line2D(x, y, transform=fig.transFigure, color=INK,
                              lw=plt.rcParams['axes.linewidth'], clip_on=False, zorder=5))

    tick, gap = 0.05, 0.035          # tick length and tick-to-label gap
    pad_x, pad_y = 0.4 * (xs[1] - xs[0]), 0.4 * (ys[0] - ys[1])

    y_ax = y_lo - fy(0.07)
    line([xs[0] - pad_x, xs[-1] + pad_x], [y_ax, y_ax])
    for x, mgsi in zip(xs, mgsi_show):
        line([x, x], [y_ax, y_ax - fy(tick)])
        fig.text(x, y_ax - fy(tick + gap), f'{mgsi:g}', ha='center', va='top',
                 fontsize=fs['col'], color=INK)
    fig.text(0.5 * (xs[0] + xs[-1]), y_ax - fy(tick + gap) - fy(0.20),
             'Mantle molar Mg/Si', ha='center', va='top', fontsize=fs['axis'], color=INK)

    x_ax = x_lo - fx(0.07)
    line([x_ax, x_ax], [ys[-1] - pad_y, ys[0] + pad_y])
    for y, diw in zip(ys, reversed(diw_show)):
        line([x_ax, x_ax - fx(tick)], [y, y])
        fig.text(x_ax - fx(tick + gap), y, f'{diw:+g}', ha='right', va='center',
                 fontsize=fs['row'], color=INK)
    # 0.42 in clears the widest tick label ('-1.3') at either font size.
    fig.text(x_ax - fx(tick + gap + 0.42), 0.5 * (ys[0] + ys[-1]),
             'Core-formation $\\Delta$IW', ha='center', va='center', rotation=90,
             fontsize=fs['axis'], color=INK)


def main(table, out, paper=False):
    df = pd.read_csv(table, comment='#')
    mgsi_show, diw_show = (PAPER_MGSI, PAPER_DIW) if paper else (MGSI_SHOW, DIW_SHOW)
    nx, ny = len(mgsi_show), len(diw_show)

    # The paper version is sized to the MNRAS text width and drops the title block -- a figure
    # in a paper is captioned by LaTeX, and repeating it above the axes wastes a third of the
    # height that the pies need. Everything else is common to both: the melt SiO2 numbers and
    # the MORB reference pie belong in the caption or the companion table, not on the grid.
    if paper:
        size = (TEXT_WIDTH_IN, 1.02 * ny + 0.95)
        fs = dict(col=7.0, row=7.0, feo=5.8, cell=5.4, axis=8.0, legend=6.5)
        pad = dict(left=0.105, right=0.930, top=0.972, bottom=0.175, wspace=0.10, hspace=0.02)
    else:
        size = (1.15 * nx + 2.6, 1.16 * ny + 1.9)
        fs = dict(col=8.5, row=8.5, feo=7.5, cell=6.6, axis=10.0, legend=8.0)
        pad = dict(left=0.090, right=0.945, top=0.885, bottom=0.175, wspace=0.10, hspace=0.02)

    # Room under the pie for the one direct label the grid still carries, on the Earth cell.
    ylim = (-1.33, 1.21)

    fig, axes = plt.subplots(ny, nx, figsize=size)
    fig.patch.set_facecolor(SURFACE)

    records = []
    for iy, diw in enumerate(reversed(diw_show)):        # oxidised at the top
        for ix, mgsi in enumerate(mgsi_show):
            ax = axes[iy][ix]
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            sel = df[np.isclose(df.mg_si, mgsi) & np.isclose(df.delta_iw, diw)]
            if sel.empty:
                continue
            row = sel.iloc[0]
            c, bad = assemblage(row)
            records.append(dict(mg_si=mgsi, delta_iw=diw, mantle_feo=row.mantle_feo,
                                melt_SiO2=row.SiO2, mass_violating=bad,
                                **{m: c.get(m, 0.0) for m in MINERALS}))

            if bad:
                # Mass-violating: the norm assigned more SiO2 than the melt contains, so the
                # assemblage is not a real composition. Show the cell as excluded rather than
                # drawing a plausible-looking pie of wrong numbers.
                ax.add_patch(Circle((0, 0), 1.0, facecolor=MUTED, edgecolor='none'))
                ax.plot([-.45, .45], [-.45, .45], color=INK_3, lw=1.4, zorder=3)
                ax.plot([-.45, .45], [.45, -.45], color=INK_3, lw=1.4, zorder=3)
                ax.set_xlim(-1.30, 1.30); ax.set_ylim(*ylim)
                ax.set_aspect('equal')
                continue

            keep = [(m, c.get(m, 0.0)) for m in MINERALS if c.get(m, 0.0) > 1e-4]
            wedges, _ = ax.pie([v for _, v in keep], colors=[COLORS[m] for m, _ in keep],
                               startangle=90, counterclock=False, radius=1.0,
                               wedgeprops=dict(edgecolor=SURFACE, linewidth=1.1))
            for wedge, (m, _) in zip(wedges, keep):
                if m in HATCHED:
                    wedge.set_hatch(HATCHED[m])
                    wedge.set_edgecolor(hatch_ink(COLORS[m]))
            # The one direct label on the grid. Melt SiO2 used to be written under every pie;
            # it is a per-cell number rather than part of the shape being compared, so it now
            # lives only in the companion CSV.
            is_earth = np.isclose(mgsi, EARTH_MANTLE_MG_SI) and np.isclose(diw, EARTH_DELTA_IW)
            if is_earth:
                ax.text(0, -1.28, 'Earth', ha='center', va='center', fontsize=fs['cell'],
                        color=INK, fontweight='bold')
            ax.set_xlim(-1.30, 1.30); ax.set_ylim(*ylim)

            if is_earth:
                ax.add_patch(Circle((0, 0), 1.16, fill=False, edgecolor=INK, lw=1.6, zorder=4))

    # Mg/Si and dIW tick labels are drawn on the axis lines in draw_grid_axes, below.
    for iy, diw in enumerate(reversed(diw_show)):
        feo = df[np.isclose(df.delta_iw, diw)].mantle_feo.iloc[0]
        axes[iy][-1].text(1.75, 0, f'{feo:.2f}', fontsize=fs['feo'], color=INK_2,
                          ha='left', va='center', transform=axes[iy][-1].transData)

    if not paper:
        fig.text(0.5, 0.962, 'Normative crust mineralogy across the composition grid',
                 ha='center', fontsize=13, color=INK)
        fig.text(0.5, 0.929,
                 'MAGEMin primary melts at 20% melting, through the CIPW norm.  '
                 'Melt SiO$_2$ per cell is in the companion CSV.',
                 ha='center', fontsize=8.5, color=INK_2)
        fig.text(0.5, 0.909,
                 'Hue = mineral family (amber quartz, green feldspar, light/dark blue Mg/Fe '
                 'pyroxene, purple melilite, red olivine); dotted = Ca-bearing.  '
                 'Akermanite and the Fe-pyroxenes carry proxied dissolution rates.',
                 ha='center', fontsize=8.5, color=INK_2)
    fig.text(0.982, 0.5, 'Mantle FeO (wt%)', va='center', rotation=-90,
             fontsize=fs['axis'] - 1.0, color=INK_2)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m],
                             edgecolor=hatch_ink(COLORS[m]) if m in HATCHED else SURFACE, lw=0.8,
                             hatch=HATCHED.get(m)) for m in MINERALS]
    labels = list(MINERALS)
    if any(r['mass_violating'] for r in records):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor='none'))
        labels.append('beyond norm')
    # Two rows once the Fe endmembers are included -- 12 entries on one row overruns the figure
    # width and shrinks the type below the 8 pt floor.
    ncol = len(labels) if len(labels) <= 7 else (len(labels) + 1) // 2
    fig.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False,
               fontsize=fs['legend'], bbox_to_anchor=(0.5, 0.002),
               handlelength=1.1, columnspacing=1.3)

    fig.subplots_adjust(**pad)
    # Drawn last: it reads panel positions, which subplots_adjust has only just fixed.
    draw_grid_axes(fig, axes, mgsi_show, diw_show, fs)

    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', facecolor=SURFACE)
    pd.DataFrame(records).to_csv(f'{out}.csv', index=False)
    print(f'wrote {out}.png / .pdf and {out}.csv ({len(records)} cells, '
          f'{sum(r["mass_violating"] for r in records)} excluded)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=CRUST_TABLE)
    ap.add_argument('--out', default=None)
    ap.add_argument('--paper', action='store_true',
                    help='MNRAS text-width version: fewer pies, no title block')
    a = ap.parse_args()
    main(a.table, a.out or ('output/crust_grid_paper' if a.paper else 'output/crust_grid'),
         paper=a.paper)
