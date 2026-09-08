"""Is oceanic crust more reactive than continental crust, and does mantle Mg/Si change that?

A DIAGNOSTIC, not part of the planet model. The model has no continental mineralogy: continental
weathering is WHAK with a rate pinned to modern Earth and a cation split fixed to modern river
chemistry, so `mantle_mg_si` and `crust_production_rate` reach only the seafloor side. That makes
the Mg/Si signal in the flux-ratio figures ambiguous -- it could be the seafloor crust genuinely
becoming more reactive, or just the continental law being unable to respond. This settles which,
without touching the model or re-running any planet.

Method
------
1. Oceanic crust: the stage-1 melt table (mantle -> basalt, F = 0.20 at 1 GPa).
2. Continental crust: the stage-2 melt of that basalt -- hydrous partial melting at 15 kbar
   leaving a garnet + cpx + amphibole residue, i.e. the TTG model of Archean continental crust
   (Rapp & Watson 1995, J. Petrol. 36, 891; Moyen & Martin 2012, Lithos 148, 312). See
   make_continental_compositions.jl.
3. BOTH assemblages go through the SAME norm (`_cipw_norm_native`) and are evaluated with the
   model's own kinetic rate law `get_k` at ONE fixed (T, pH).

Two choices that matter, and why
--------------------------------
* One norm for both sides. The model itself uses the pyrolite norm (`cipw_norm`) for oceanic
  crust, but that norm REFUSES a corundum-normative rock, and the stage-2 melts are peraluminous
  (Al2O3 ~21 wt% against CaO ~6 + Na2O ~6), which is normal for felsic melts. `_cipw_norm_native`
  accepts them, dropping the excess Al with a warning. Using the model's norm for the numerator
  and the native one for the denominator would put a 1.26x norm artefact straight into the ratio
  -- measured at the Earth point, k_oceanic is 2.92e-10 under pyrolite and 3.68e-10 under native.
  So both sides use the native norm and the ratio is internally consistent.
* One fixed (T, pH) for both. Evaluating each crust at its own equilibrium pH would fold a
  chemistry difference into what is meant to be a reactivity ratio.

What this can and cannot settle
-------------------------------
`get_k` is the kinetic term only -- no transport, no supply limitation, no climate. That is
deliberate: it isolates the one thing the planet model holds frozen. The ABSOLUTE ratio is
sensitive to the stage-2 melting parameters and to the norm; the NORMALISED map (relative to the
Earth-like crust) is the robust product, because those choices largely cancel. A flat map would
mean Mg/Si does not act through intrinsic reactivity, NOT that Mg/Si is unimportant.
"""

import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from kamino.chemistry import get_k, alk_idx
from kamino.crust_composition import _cipw_norm_native
from kamino.constants import EARTH_MANTLE_MG_SI, EARTH_DELTA_IW

import plot_results as pr

# The single condition both crusts are compared at. Edit these rather than passing flags.
REF_T = 288.15      # K
REF_PH = 7.0
REF_P = 1e5         # Pa; get_k does not use it, passed for signature compatibility

REF_MG_SI = float(EARTH_MANTLE_MG_SI)
REF_DIW = float(EARTH_DELTA_IW)

OXIDES = ('SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O')

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, '../src/kamino/data')
OCEANIC_CSV = os.path.join(DATA, 'crust_compositions.csv')
CONTINENTAL_CSV = os.path.join(DATA, 'continental_compositions.csv')


def _load(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nGenerate the {what} table first.")
    return pd.read_csv(path, comment='#')


def _oxides(row):
    return {o: float(row[o]) for o in OXIDES if o in row}


def _norm(oxides):
    """Native CIPW norm, returning (composition, peraluminous_excess_or_0).

    Warnings are captured rather than printed: peraluminous melts are the expected case here,
    and 650 identical warnings would bury anything that actually matters.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp = _cipw_norm_native(oxides, emit_quartz=True)
    excess = 0.0
    for w in caught:
        msg = str(w.message)
        if 'peraluminous' in msg:
            try:
                excess = float(msg.split(':')[1].split('mol')[0])
            except (IndexError, ValueError):
                excess = np.nan
    return comp, excess


def alk_rate(composition, T=REF_T, pH=REF_PH):
    """Alkalinity kinetic rate constant for one mineral assemblage, at fixed T and pH."""
    return float(get_k(REF_P, T, pH, composition)[alk_idx])


def reactivity_table(oc_df, ct_df, T=REF_T, pH=REF_PH, report=True):
    """Per grid point: oceanic and continental alkalinity rate constants and their ratio."""
    oc_by_key = {(round(float(r['mg_si']), 6), round(float(r['delta_iw']), 6)): r
                 for _, r in oc_df.iterrows()}
    rows, skipped = [], Counter()
    for _, cr in ct_df.iterrows():
        key = (round(float(cr['mg_si']), 6), round(float(cr['delta_iw']), 6))
        orow = oc_by_key.get(key)
        if orow is None:
            skipped['no matching oceanic row'] += 1
            continue
        try:
            oceanic, _ = _norm(_oxides(orow))
            continental, excess = _norm(_oxides(cr))
        except Exception as e:                       # surfaced, not swallowed
            skipped[f'{type(e).__name__}: {str(e)[:60]}'] += 1
            continue
        k_oc, k_ct = alk_rate(oceanic, T, pH), alk_rate(continental, T, pH)
        if not (np.isfinite(k_oc) and np.isfinite(k_ct)) or k_oc <= 0 or k_ct <= 0:
            skipped['non-positive or non-finite rate'] += 1
            continue
        rows.append({'mg_si': key[0], 'delta_iw': key[1], 'k_oceanic': k_oc,
                     'k_continental': k_ct, 'ratio': k_oc / k_ct,
                     'peraluminous_excess': excess})
    if report and skipped:
        print("  skipped:")
        for reason, n in skipped.most_common():
            print(f"    {n:4d}  {reason}")
    return pd.DataFrame(rows)


def _earth_value(tab):
    hit = tab[np.isclose(tab['mg_si'], REF_MG_SI) & np.isclose(tab['delta_iw'], REF_DIW)]
    return float(hit['ratio'].iloc[0]) if len(hit) else np.nan


def _sensitivity(oc_df, ct_df):
    """How much does the reference condition move the answer? Reported, not hidden."""
    oc = oc_df[np.isclose(oc_df.mg_si, REF_MG_SI) & np.isclose(oc_df.delta_iw, REF_DIW)]
    ct = ct_df[np.isclose(ct_df.mg_si, REF_MG_SI) & np.isclose(ct_df.delta_iw, REF_DIW)]
    print("\nEarth-point ratio against the reference condition:")
    print(f"  {'T (K)':>7} {'pH':>5} {'k_oceanic':>12} {'k_continental':>14} {'ratio':>8}")
    for T in (278.15, 288.15, 298.15):
        for pH in (6.0, 7.0, 8.0):
            t = reactivity_table(oc, ct, T, pH, report=False)
            if t.empty:
                continue
            r = t.iloc[0]
            print(f"  {T:7.2f} {pH:5.1f} {r['k_oceanic']:12.4g} "
                  f"{r['k_continental']:14.4g} {r['ratio']:8.3f}")


def plot_reactivity_map(tab, output_path, normalise=True, step=0.05):
    """Contour map of the oceanic/continental kinetic ratio over Mg/Si x dIW."""
    earth = _earth_value(tab)
    if normalise and not np.isfinite(earth):
        print("No Earth reference point -- cannot draw the normalised map.")
        return
    z = (tab['ratio'] / earth) if normalise else tab['ratio']
    label = ('Seafloor / continental reactivity\n(relative to Earth-like crust)' if normalise
             else 'Seafloor / continental\nalkalinity rate constant')

    mg_vals = np.array(sorted(tab['mg_si'].unique()))
    dw_vals = np.array(sorted(tab['delta_iw'].unique()))
    Z = np.full((len(dw_vals), len(mg_vals)), np.nan)
    mi = {v: i for i, v in enumerate(mg_vals)}
    di = {v: i for i, v in enumerate(dw_vals)}
    for (_, r), v in zip(tab.iterrows(), z):
        if np.isfinite(v) and v > 0:
            Z[di[r['delta_iw']], mi[r['mg_si']]] = np.log10(v)
    if not np.isfinite(Z).any():
        print("Nothing finite to contour -- skipping the map.")
        return
    Zm = np.ma.masked_invalid(Z)

    lo = np.floor(np.nanmin(Z) / step) * step
    hi = np.ceil(np.nanmax(Z) / step) * step
    if not (hi > lo):
        lo, hi = lo - step, hi + step
    bands = np.arange(lo, hi + 0.5 * step, step)
    centre = 0.0 if normalise else np.log10(earth) if np.isfinite(earth) else 0.0
    norm = pr.mcolors.TwoSlopeNorm(vmin=min(lo, centre - step), vcenter=centre,
                                   vmax=max(hi, centre + step))

    fig, ax = pr.plt.subplots(1, 1, figsize=pr.figure_size('single', height=3.0))
    cf = ax.contourf(mg_vals, dw_vals, Zm, levels=bands, cmap=pr.cmr.fusion_r, norm=norm,
                     extend='both')
    if np.nanmin(Z) < centre < np.nanmax(Z):
        cs = ax.contour(mg_vals, dw_vals, Zm, levels=[centre], colors='k', linewidths=1.4)
        ax.clabel(cs, fmt={centre: 'Earth-like' if normalise else 'equal'}, fontsize=7,
                  inline=True)
    ax.scatter(REF_MG_SI, REF_DIW, marker='*', s=170, color='gold', edgecolors='k',
               linewidths=0.7, zorder=6)
    ax.set_xlabel('Mantle Mg/Si')
    ax.set_ylabel('Core-formation $\\Delta$IW')

    # Tick at round RATIOS, not at round powers of ten. Ticking the log axis uniformly gives
    # labels like 0.562341 and 1.77828, which are exact and unreadable.
    nice = [0.1, 0.2, 0.25, 0.33, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10]
    shown = [v for v in nice if lo <= np.log10(v) <= hi]
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=24,
                        ticks=[np.log10(v) for v in shown])
    cbar.set_label(label)
    cbar.set_ticklabels([f'{v:g}' for v in shown])
    cbar.ax.axhline(centre, color='k', linewidth=1.2)
    stem = 'crust_reactivity_ratio' + ('' if normalise else '_absolute')
    pr._save_fig(fig, pr.figure_path(output_path, f'{stem}.png'))


if __name__ == '__main__':
    out = pr.DEFAULT_OUTPUT_PATH
    oc_df = _load(OCEANIC_CSV, 'oceanic (make_crust_compositions.jl)')
    ct_df = _load(CONTINENTAL_CSV, 'continental (make_continental_compositions.jl)')
    print(f"oceanic melts: {len(oc_df)}   continental melts: {len(ct_df)}")

    tab = reactivity_table(oc_df, ct_df)
    print(f"reactivity at T = {REF_T:.2f} K, pH = {REF_PH:.1f} for {len(tab)} point(s)")

    n_per = int((tab['peraluminous_excess'] > 0).sum())
    if n_per:
        print(f"  {n_per}/{len(tab)} continental melts peraluminous; excess Al dropped, "
              f"max {tab['peraluminous_excess'].max():.3g} mol")

    earth = _earth_value(tab)
    print(f"\nEarth reference (Mg/Si {REF_MG_SI:g}, dIW {REF_DIW:g}): "
          f"seafloor/continental = {earth:.3f}")
    lo_r, hi_r = tab['ratio'].min(), tab['ratio'].max()
    print(f"across the grid: {lo_r:.3g} to {hi_r:.3g} "
          f"({lo_r / earth:.2f}x to {hi_r / earth:.2f}x Earth's)")

    print(f"\nRatio by Mg/Si at dIW = {REF_DIW:g}:")
    cut = tab[np.isclose(tab['delta_iw'], REF_DIW)].sort_values('mg_si')
    for _, r in cut.iloc[::3].iterrows():
        print(f"  Mg/Si {r['mg_si']:4.2f}   ratio {r['ratio']:7.3f}   "
              f"{r['ratio'] / earth:6.2f}x Earth")

    print(f"\nRatio by dIW at Mg/Si = {REF_MG_SI:g}:")
    cut = tab[np.isclose(tab['mg_si'], REF_MG_SI)].sort_values('delta_iw')
    for _, r in cut.iloc[::4].iterrows():
        print(f"  dIW {r['delta_iw']:+5.2f}   ratio {r['ratio']:7.3f}   "
              f"{r['ratio'] / earth:6.2f}x Earth")

    plot_reactivity_map(tab, out, normalise=True)
    plot_reactivity_map(tab, out, normalise=False)
    _sensitivity(oc_df, ct_df)
    tab.to_csv(os.path.join(out, 'crust_reactivity.csv'), index=False)
    print(f"\nSaved {os.path.join(out, 'crust_reactivity.csv')}")
