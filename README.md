# zero-am-horseshoe

Code and result files for the computer-assisted proofs in

> Jaeseo Lee, *A computer-assisted horseshoe in the planar equal-mass three-body problem with zero angular momentum*.

The paper proves, for the planar Newtonian three-body problem with masses 1, G = 1, zero angular momentum and energy −1/2:

- **Theorem 1 (topological horseshoe).** The map between consecutive collinear configurations has a compact invariant set on which every mutual distance stays in [0.217857228763, 7.419986043806], consecutive collinear configurations are at least 0.276707990268 apart in time, and the map is semiconjugate to a subshift with entropy log ρ ≥ 0.0073696936384425343859428.
- **Theorem 2 (uniform hyperbolicity and conjugacy).** With cone conditions, this set is uniformly hyperbolic and the map is conjugate to the vertex shift of a finite graph.

Other negative energies follow by scaling. Both theorems rest on 314 covering relations between 313 h-sets and on cone conditions, which were certified by two programs with independent source code. This repository contains both programs, every result file that the proofs and the numbers of the paper depend on, and the checks that were run around them.

A note on language and names: code comments, docstrings and the `tag` strings inside the JSON files are in Korean. Directory names (`T3-cap`, `T6-horseshoe`, …) are the track names used during the research. Names such as BRIEF, R1, E1, F1, G5–G9 or V2 in comments and records are item names of the private working notes of the project; they are kept because the code and result files are fixed by the SHA-256 hashes in the manifests.

## Layout

| Path | Role in the paper |
|---|---|
| `results/2026-10-15-T6-horseshoe/graph_design.json` | **Definition** of the 313 h-sets (centres and frames as hexadecimal doubles, radii as double-precision numbers; both programs convert them exactly to rationals) and of the graph with 314 edges |
| `tracks/T3-cap/` | First program: C++ engine on CAPD (`engine/t3cap_engine.cpp`), build scripts, Python API and exact judgement (`t3cap/`) |
| `tracks/T6-horseshoe/` | First program: covering decisions and assembly of Theorem 1 |
| `tracks/T7-hyperbolic/` | First program: cone decisions, the second cone method (Weyl inequality and exact Cholesky), assembly of Theorem 2 |
| `results/2026-10-17-T6-horseshoe/coverings/` | First program: the 314 stored covering records of the strict build (`-ffp-contract=off`) |
| `results/2026-10-17-T6-horseshoe/theorem_A.json`, `results/2026-10-17-T7-hyperbolic/theorem_Aprime.json` | First program: the constants of Theorems 1 and 2 and their exact decisions |
| `tracks/T11-arbcert/` | Second program: Arb ball arithmetic through python-flint, two Taylor engines, exact judgement and assembly |
| `results/2026-10-17-T11-arbcert/` | Second program: per-edge records (`tubes/`, `stage2/`), `exact_judge.json`, `theorem_claims.json`, controls, cross-check with the first program |
| `tracks/V2-compact/`, `V3-cone/`, `V5-indep/`, `V7-judge/` | Only the 20 original files from which parts of the second program were copied; `t11cli.py provenance check` compares them with the copies and the recorded diffs |
| `results/2026-10-15-*`, `results/2026-10-16-T7-hyperbolic/` | Records of the earlier build (without `-ffp-contract=off`), kept for the bitwise comparison with the strict build, and the figure-eight test case. **The records the theorems rest on are those in the `2026-10-17` folders.** |
| `results/2026-10-15-T2-num/`, `results/2026-10-15-T5-geom/` | Three files of the numerical design stage (periodic orbit, homoclinic point, curvature survey) quoted in the paper; the code that produced them is not part of this repository, and the proofs do not use the existence of these orbits |
| `tracks/PF-figures/`, `paper/p1/figures/` | Figures of the paper |
| `CONSTANTS.tsv` | Every number typeset in the paper → result file and JSON pointer |

Each results folder has a `manifest.json` with the SHA-256 of its result files and of the code they depend on; the checks below compare them. The `git_sha` fields in the manifests refer to the private working repository in which the computations were run. This repository starts from a fresh history. Three result files were edited for this release, and the corresponding hashes in the manifests were updated: in `results/2026-10-1{5,7}-T3-cap/capd_selftests.json` the absolute path of the working copy in a CTest log excerpt was replaced by `<repo>/`, and in `results/2026-10-17-T11-arbcert/crossing_jets_mutant.json` a temporary directory in a recorded command line was replaced by `<tmp>/`. None of these strings enters a decision. Apart from these three files, the three manifests that list them and the comment in `.gitattributes`, every file taken from the working repository is byte-identical to it.

## Requirements

- Python 3.12 with the packages in `requirements.lock.txt` (python-flint 0.9.0, mpmath 1.3.0, numpy, scipy, matplotlib, pytest). The scripts call `.venv/bin/python` relative to the repository root:

  ```bash
  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
  ```

- Only for path (b), first program: macOS on arm64 with the Xcode command line tools (`/usr/bin/clang++`), CMake, MPFR 4.2.2 and GMP from Homebrew under `/opt/homebrew`, and Rosetta 2 for the x86_64 build with filib intervals. The build scripts have these paths fixed and have not been tested on other platforms.

The scripts limit themselves to three worker processes and use `nice -n 10`; set one BLAS thread as below.

## (a) Re-deciding the stored results without building anything

The engines return interval endpoints as exact hexadecimal doubles or outward-rounded decimal strings, and every inequality is decided in exact rational arithmetic in Python. The commands below re-read the stored enclosures and repeat the decisions. None of them needs CAPD or a compiler, and none writes into the repository. Times were measured on an arm64 laptop.

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
```

First program (stored CAPD enclosures):

| Command | Checks | Time |
|---|---|---|
| `$PY tracks/T6-horseshoe/scripts/check_results.py` | 314 coverings re-decided, manifest hashes | 2 s |
| `$PY tracks/T6-horseshoe/scripts/assemble_theorem.py --no-write` | hypotheses of Theorem 1 and the constants δ, R, t_min, ρ, log ρ | 5 s |
| `$PY tracks/T6-horseshoe/scripts/stored_controls.py --no-write` | 942 negative controls on the stored enclosures | 1 s |
| `$PY tracks/T6-horseshoe/scripts/compare_strict_rebuild.py --no-write` | earlier and strict build, bitwise comparison | 1 s |
| `$PY tracks/T7-hyperbolic/scripts/check_results.py` | stored cone results, manifest hashes | 2 s |
| `$PY tracks/T7-hyperbolic/scripts/certify_cones.py --no-write` | cone conditions on 314 edges, 942 controls | 2 s |
| `$PY tracks/T7-hyperbolic/scripts/assemble_aprime.py --no-write` | constants of Theorem 2 | 5 s |
| `$PY tracks/T7-hyperbolic/scripts/g8_cones.py --no-write` | second cone method (942 decisions) and the x86_64 filib rerun | 2 s |
| `PYTHONPATH=tracks/T3-cap:tracks/T3-cap/scripts $PY tracks/T3-cap/scripts/check_results.py` | stored engine results and manifest hashes | 1 s |
| `(cd tracks/T6-horseshoe && ../../$PY -m pytest -q -p no:cacheprovider tests)` | 64 specification tests | 1 s |
| `(cd tracks/T7-hyperbolic && ../../$PY -m pytest -q -p no:cacheprovider tests)` | 39 specification tests | 1 s |

Second program (stored Arb enclosures):

```bash
T11="$PY -I -B tracks/T11-arbcert/t11cli.py"; TMP=$(mktemp -d); RES=results/2026-10-17-T11-arbcert
$T11 provenance check && $T11 manifest check && $T11 jobs status --strict
$PY -I -B tracks/T11-arbcert/exact_judge.py --out $TMP/exact_judge.json
$T11 assemble --out $TMP/theorem_claims.json --merged-tubes $TMP/rigorous_tubes.json
$PY -I -B tracks/T11-arbcert/graph_and_hulls.py --out $TMP/graph_and_hulls.json
$T11 compare $RES/exact_judge.json $TMP/exact_judge.json $RES/theorem_claims.json $TMP/theorem_claims.json \
             $RES/rigorous_tubes.json $TMP/rigorous_tubes.json $RES/graph_and_hulls.json $TMP/graph_and_hulls.json
$T11 controls --out $TMP/controls.json && $T11 compare $RES/controls.json $TMP/controls.json
$T11 g8 --capd-dir results/2026-10-17-T6-horseshoe/coverings --out $TMP/g8.json \
  && $T11 compare $RES/g8_capd_crosscheck_2026-10-17.json $TMP/g8.json
```

Each of these takes under a second except `controls` (1884 edge-level and 47 theorem-level controls, about 14 s). `bash tracks/T11-arbcert/verify.sh` runs all of them and in addition re-integrates three sample points, reruns two edges under an audit hook that fails on any access to the first program's code or records, and runs the specification tests (about 2 minutes in total). It needs no compiled code.

Paper numbers and figures:

| Command | Checks | Time |
|---|---|---|
| `$PY -I -B tracks/T8-synthesis/scripts/build_derived.py --check` | aggregate numbers (sums, minima over edges, running times) recomputed from the result files | under 1 s |
| `bash tracks/PF-figures/verify.sh` | figures regenerated in a temporary folder, `figdata.json` compared | 12 s |

## (b) Certifying from scratch

**Second program (no compiled code).** `bash tracks/T11-arbcert/run_all.sh` recomputes all 314 flow tubes and box enclosures (about 17 minutes of wall time in two lanes), the exact decisions, the controls, the sampling checks (several hours) and the manifest. It writes into `results/2026-10-17-T11-arbcert/`, so run it in a separate copy of the repository. Its step `tube-rerun` compares with an audit record that is not included here and will fail; that step is a reproduction check and not an input of the certification.

**First program (CAPD).**

```bash
bash tracks/T3-cap/fetch_capd.sh                    # CAPD at the pinned commit, tree hash checked
bash tracks/T3-cap/build_capd.sh native-strict       # arm64, -ffp-contract=off, MPFR
bash tracks/T3-cap/build_capd.sh x86filib-strict     # x86_64 under Rosetta 2, filib intervals
bash tracks/T3-cap/build_engine.sh native-strict
bash tracks/T3-cap/build_engine.sh x86filib-strict
$PY tracks/T3-cap/scripts/check_build_flags.py       # no fused multiply-add in the strict binaries
$PY tracks/T6-horseshoe/scripts/certify_graph.py --out-dir <dir> --no-write   # all 314 edges
```

The engine runs took 15,743.5 s summed over the edges, with three engines in parallel. `tracks/T3-cap/verify.sh`, `tracks/T6-horseshoe/verify.sh` and `tracks/T7-hyperbolic/verify.sh` run the interval self-tests, the compiler probe, the specification tests, the mutation tests and a two-edge engine reproduction; `bash scripts/verify_all.sh` runs every `verify.sh`.

## (c) Where each number comes from

`CONSTANTS.tsv` lists the 115 keys of the numbers typeset in the paper. For each key it gives the printed value, the rounding direction (`lower`: the true value is at least the printed value; `upper`: at most; `equal`: exact; `estimate`: a descriptive value rounded to nearest and not used as a bound), the result file and JSON pointer it comes from (for a few keys an exact sum, difference, quotient or square root of such entries), and a short description.

The main constants:

| Paper | Value | Direction | Source |
|---|---|---|---|
| h-sets, edges | 313, 314 | exact | `graph_design.json#/summary/vertices`, `theorem_Aprime.json#/edges` |
| δ | 0.217857228763 | lower | `theorem_A.json#/collision_distance_lower_bound_delta` |
| R | 7.419986043806 | upper | `theorem_A.json#/size_upper_bound_R` |
| t_min | 0.276707990268 | lower | `theorem_A.json#/return_time_lower_bound` |
| ρ | 1.007396916664634 | lower | `theorem_A.json#/rho_lower_bound` |
| log ρ | 0.0073696936384425343859428 | lower | `theorem_A.json#/claimed/ln_rho` |
| μ_s, μ_u | 0.8944, 1.1167 | upper, lower | `theorem_Aprime.json` |
| ε, λ_s, λ_u, c_s, c_u | 0.1056, 0.9458, 1.0567, 3.078, 3.252 | see `CONSTANTS.tsv` | `theorem_Aprime.json` |
| covering margin, second program | 0.061029134539 | lower | `results/2026-10-17-T11-arbcert/theorem_claims.json#/measured/cover_margin/exact` |

`theorem_A.json` is in `results/2026-10-17-T6-horseshoe/` and `theorem_Aprime.json` in `results/2026-10-17-T7-hyperbolic/`.

## What needs a build

| Part of the proof | From stored files | Needs the CAPD build |
|---|---|---|
| Exact decisions of the covering, cone, separation and constant inequalities | yes, path (a) | no |
| Enclosures of the flow, first program | no | yes |
| Enclosures of the flow, second program | no, but no compiled code (`run_all.sh`) | no |
| ρ, log ρ and λ_s, λ_u, c_s, c_u from μ_s, μ_u | yes, exact arithmetic | no |

The sampling checks against mpmath reference solutions and the comparison of the two programs are consistency checks and not part of the proofs.

## CAPD

CAPD is not included. `tracks/T3-cap/fetch_capd.sh` downloads commit `2f060980d0685cc9bf88352e08c242197cd7d686` (version 6.1.0-dev, no release tag) of

- upstream: https://github.com/CAPDGroup/CAPD
- preserved fork with the same commit: https://github.com/Chocothin/CAPD (tag `zero-am-horseshoe-pin` points to that commit)

and checks the tree hash recorded in `tracks/T3-cap/third_party/CAPD_SOURCE.txt`. If the commit is no longer available upstream, clone the fork, check out that commit, copy the tree without `.git` to `tracks/T3-cap/third_party/capd/`, and run `fetch_capd.sh` again; it then only compares the tree hash.

## License

GNU General Public License v3.0 (see `LICENSE`), because the engine links to CAPD, which is distributed under GPL-3.0. The license applies to the code and to the result files.

## Citation

Jaeseo Lee, *A computer-assisted horseshoe in the planar equal-mass three-body problem with zero angular momentum*. Tag `paper1-v1` marks the state of this repository that accompanies the paper.
