#=
make_continental_compositions.jl -- a second melting stage: oceanic crust -> continental crust.

This is a DIAGNOSTIC, not part of the planet model. It exists to answer one question: does
changing mantle Mg/Si make the oceanic crust more reactive RELATIVE to the continental crust it
would produce? The planet model has no continental mineralogy and is not gaining one; see
experiments/crust_reactivity.py for the comparison this table feeds.

Mechanism: partial melting of hydrous metabasalt, keeping the melt and discarding the residue.
That is the TTG (tonalite-trondhjemite-granodiorite) model of Archean continental crust, the
standard account of how a basaltic parent yields a felsic daughter -- experimentally calibrated
by Rapp & Watson (1995), J. Petrol. 36, 891 (dehydration melting of metabasalt, 8-32 kbar,
TTG melt in equilibrium with a garnet + clinopyroxene +/- rutile residue), and reviewed by
Moyen & Martin (2012), Lithos 148, 312.

It is deliberately the ARCHEAN mechanism. Modern bulk continental crust is andesitic and is
made by arc magmatism plus foundering of dense lower-crustal cumulates (Rudnick 1995, Nature 378,
571; Jagoutz & Kelemen 2015, Annu. Rev. Earth Planet. Sci. 43, 363), which one melting step
cannot reproduce. For an abiotic, reducing planet an Archean-style crust is the self-consistent
choice, and TTG is low-K by definition, which suits a model that deliberately carries no K.

Input:  crust_compositions.csv   (stage 1: mantle -> oceanic crust, F = 0.20 at 1.0 GPa)
Output: continental_compositions.csv  (stage 2: oceanic crust -> continental crust)

Usage:  julia --project=... src/kamino/data/make_continental_compositions.jl [out.csv]
=#

using MAGEMin_C, Printf

const OX = ["SiO2","Al2O3","CaO","MgO","FeO","K2O","Na2O","TiO2","O","Cr2O3","H2O"]
const CSV_OXIDES = ["SiO2","TiO2","Al2O3","Cr2O3","FeO","MgO","CaO","Na2O","K2O"]

# --- Stage-2 parameters ------------------------------------------------------------------------
# These three are CHOICES, not inheritances, and the result is most sensitive to the water.
#
# P: garnet must be residual for a TTG-like melt, which needs ~>1.5 GPa. Rapp & Watson span
#    8-32 kbar; 15 kbar sits in the garnet-amphibolite/eclogite transition they sample.
# H2O: dry basalt melting does NOT give TTG. 3 wt% is a mid-range hydrated-basalt value; the
#    anhydrous oxides are renormalised to (100 - H2O) so the ratio between them is preserved.
# F: TTG melt fractions are typically 0.1-0.4. 0.20 matches stage 1, leaving one fewer
#    difference between the two stages to account for.
const P_MELT2 = 15.0        # kbar
const H2O_WT = 3.0          # wt% added to the basalt before melting
const F_TARGET2 = 0.20
const T_LO, T_HI, T_TOL = 600.0, 1400.0, 1.0   # hydrous basalt solidus is ~700 C at 15 kbar
const F_TOL = 0.05

minim(data, X, T) = single_point_minimization(P_MELT2, T, data, X=X, Xoxides=OX, sys_in="wt")

"Liquid composition in wt% over CSV_OXIDES, plus melt fraction. (nothing, 0) if there is no melt."
function melt_oxides(out)
    findfirst(==("liq"), out.ph) === nothing && return nothing, 0.0
    frac = out.frac_M_wt
    (frac === nothing || isnan(frac) || frac <= 0) && return nothing, 0.0
    comp = Dict(OX[j] => out.bulk_M_wt[j] * 100 for j in eachindex(OX))
    kept = Dict(o => get(comp, o, 0.0) for o in CSV_OXIDES)
    tot = sum(values(kept))
    tot <= 0 && return nothing, frac
    return Dict(o => v / tot * 100 for (o, v) in kept), frac
end

residue(out) = join(filter(!=("liq"), out.ph), ";")

"Temperature giving F_TARGET2 at P_MELT2, by bisection. NaN if unreachable below T_HI."
function T_for_F(data, X)
    melt_oxides(minim(data, X, T_HI))[2] < F_TARGET2 && return NaN
    lo, hi = T_LO, T_HI
    while hi - lo > T_TOL
        mid = (lo + hi) / 2
        melt_oxides(minim(data, X, mid))[2] < F_TARGET2 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"Bulk X for MAGEMin from a stage-1 oceanic-crust analysis, hydrated to H2O_WT."
function hydrated_basalt(row::Dict{String,Float64})
    dry = sum(row[o] for o in CSV_OXIDES)
    s = (100.0 - H2O_WT) / dry
    get2(o) = get(row, o, 0.0) * s
    return [get2("SiO2"), get2("Al2O3"), get2("CaO"), get2("MgO"), get2("FeO"),
            get2("K2O"), get2("Na2O"), get2("TiO2"), 0.0, get2("Cr2O3"), H2O_WT]
end

# --- Grid --------------------------------------------------------------------------------------

const CSV_HEADER = "mg_si,delta_iw,T_melt2,melt_fraction2,residual_phases2," *
                   join(CSV_OXIDES, ",")

"Read stage 1, skipping its comment banner."
function read_stage1(path)
    lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(path))
    header = split(lines[1], ",")
    rows = Vector{Tuple{Float64,Float64,Dict{String,Float64}}}()
    for l in lines[2:end]
        f = split(l, ",")
        d = Dict{String,Float64}()
        for o in CSV_OXIDES
            j = findfirst(==(o), header)
            j === nothing && error("stage 1 is missing oxide $o")
            d[o] = parse(Float64, f[j])
        end
        mgsi = parse(Float64, f[findfirst(==("mg_si"), header)])
        diw = parse(Float64, f[findfirst(==("delta_iw"), header)])
        push!(rows, (mgsi, diw, d))
    end
    return rows
end

function main()
    here = dirname(@__FILE__)
    inpath = joinpath(here, "crust_compositions.csv")
    outpath = length(ARGS) >= 1 ? ARGS[1] : joinpath(here, "continental_compositions.csv")

    src = read_stage1(inpath)
    @printf("stage 2: %d points from %s\n", length(src), basename(inpath))
    @printf("  P = %.1f kbar, H2O = %.1f wt%%, F = %.2f\n", P_MELT2, H2O_WT, F_TARGET2)
    flush(stdout)

    data = Initialize_MAGEMin("ig", verbose=false)
    rows, failures = String[CSV_HEADER], String[]
    t0 = time()
    for (i, (mgsi, diw, ox)) in enumerate(src)
        row, status = try
            X = hydrated_basalt(ox)
            T = T_for_F(data, X)
            if isnan(T)
                (nothing, @sprintf("cannot reach F=%.2f below %.0f C", F_TARGET2, T_HI))
            else
                out = minim(data, X, T)
                m, F = melt_oxides(out)
                if m === nothing
                    (nothing, @sprintf("no melt at %.0f C", T))
                else
                    w = abs(F - F_TARGET2) > F_TOL ? @sprintf("F=%.3f off target", F) : "ok"
                    (@sprintf("%.4g,%.4g,%.0f,%.6g,%s,", mgsi, diw, T, F, residue(out)) *
                     join([@sprintf("%.6g", m[o]) for o in CSV_OXIDES], ","), w)
                end
            end
        catch e
            (nothing, "EXCEPTION $(typeof(e))")
        end
        if row === nothing
            push!(failures, @sprintf("Mg/Si=%.2f dIW=%+.1f: %s", mgsi, diw, status))
        else
            push!(rows, row)
        end
        if i % 50 == 0
            @printf("  %d/%d  (%.0f s, %d failed)\n", i, length(src), time() - t0,
                    length(failures))
            flush(stdout)
        end
    end

    open(outpath, "w") do io
        println(io, "# Generated by make_continental_compositions.jl -- do not edit by hand.")
        println(io, "# Stage 2: hydrous partial melting of the stage-1 oceanic crust (TTG model).")
        @printf(io, "# isobaric %.1f kbar, H2O=%.1f wt%%, F_TARGET=%.2f, batch melting\n",
                P_MELT2, H2O_WT, F_TARGET2)
        println(io, "# Melt is the continental crust; the residue is discarded.")
        for r in rows
            println(io, r)
        end
    end
    @printf("wrote %s (%d rows, %d failed, %.0f s)\n", outpath, length(rows) - 1,
            length(failures), time() - t0)
    for f in failures[1:min(end, 15)]
        println("  FAILED ", f)
    end
    length(failures) > 15 && @printf("  ... and %d more\n", length(failures) - 15)
end

main()
