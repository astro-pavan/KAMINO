"""Re-run the wall_timeout runs that feed the ratio figures, with a longer wall budget.

One-off. The reusable part -- KAMINO_RERUN and KAMINO_WALL_SHALLOW reaching spawned workers --
is a permanent change in parameter_sweep.py.

The __main__ guard is load-bearing: ProcessPoolExecutor uses spawn, so every worker re-imports
this module. Without the guard each worker would re-execute the launch and try to start its own
pool, which multiprocessing detects and turns into a BrokenProcessPool.
"""
import os
import sys

import numpy as np

import parameter_sweep as ps
import continental_baseline as cb
import plot_results as pr

OUT = '/home/pt426/Code/kamino/sweep_output'


def _int(v):
    """Ints where the run name needs them, so a rebuilt combo reproduces its original filename."""
    return int(v) if float(v).is_integer() else float(v)


def wall_timeout_combos(df):
    w = df[df.termination == 'wall_timeout']
    rel = w[(w.ocean_depth == 3000) & (w.f_HT == 0.0) & np.isclose(w.pe, -3.0)
            & np.isclose(w.kd_mg, cb.KD_MG_CALIB) & np.isclose(w.k_na, cb.K_NA_CALIB)
            & w.instellation.isin(cb.COARSE_INSTELLATION)
            & w.land_fraction.apply(lambda v: any(np.isclose(v, l)
                                                  for l in cb.COARSE_LAND_FRACTIONS))
            & w.outgassing.isin(cb.GRID_OUTGASSING)
            & w.crust_production.isin(cb.GRID_CRUST)
            & w.mg_si.apply(lambda v: any(np.isclose(v, m) for m in cb.GRID_MG_SI))]
    return [(float(r.instellation), _int(r.outgassing), _int(r.crust_production),
             _int(r.ocean_depth), bool(r.reverse_weathering), float(r.mg_si),
             float(r.delta_iw), float(r.alpha), float(r.kd_mg), float(r.k_na),
             float(r.pe), float(r.land_fraction))
            for _, r in rel.iterrows()]


if __name__ == '__main__':
    assert ps.RERUN, "KAMINO_RERUN must be set, or existing wall_timeout output is just reused"
    print(f"RERUN={ps.RERUN}  wall budget (shallow) = {ps.WALL_SECONDS_SHALLOW} s", flush=True)

    combos = wall_timeout_combos(pr.load_data(OUT))
    names = [cb._run_name(*c) for c in combos]
    on_disk = sum(os.path.exists(os.path.join(OUT, f'{n}.json')) for n in names)
    print(f"{len(combos)} wall_timeout runs; {on_disk} matched their existing filename")
    if on_disk != len(combos):
        for n in [n for n in names if not os.path.exists(os.path.join(OUT, f'{n}.json'))][:5]:
            print("   not matched:", n)
        sys.exit("aborting: run names do not round-trip, fix _int() first")

    combos.sort(key=lambda c: (ps._cost_rank(c), c[11] == 0.0))
    cb.run(combos, output_path=OUT)
