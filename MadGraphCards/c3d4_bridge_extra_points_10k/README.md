# MadGraph `gg -> hhhh` c3/d4 bridge points

This directory records the exact MadGraph setup observed in the active
Odysseus extra-point campaign on 2026-07-11.  The campaign was running from

```text
/home/apapaefs/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_4h_c3d4
```

with

```text
./run_c3d4_bridge_10k.sh --cores 192
```

At capture time (`2026-07-11T11:01:06-04:00`) the active subprocess was

```text
bin/generate_events run_gg_4h_4_12.0_0.0 -f --multicore --nb_core=192
```

The process is loop-induced `g g > h h h h [noborn=QCD]`, generated with
MadGraph5_aMC@NLO 3.5.15 and the `loop_sm_c3d4` model.  The run card requests
10,000 unweighted events per point at 14 TeV, uses `nn23lo1`/LHAPDF ID 230000,
and leaves the renormalization and factorization scales dynamical.

## What each file is

- `run_c3d4_bridge_10k.sh`: resumable campaign driver.  It edits only the c3
  and d4 entries and `nevents`, runs points sequentially, uses MadGraph
  multicore parallelism within a point, validates the LHE event count and
  embedded couplings, and writes a CSV manifest.
- `Cards/proc_card_mg5.dat`: process-generation card for the loop-induced
  `gg -> hhhh` process and the `loop_sm_c3d4` model.
- `Cards/param_card_template.dat`: pre-campaign parameter-card backup.  The
  driver uses this card as its mutable template and replaces TRIPCOUP entry 4
  (c3) and QUARTCOUP entry 6 (d4) for every point.
- `Cards/run_card_template.dat`: pre-campaign run-card backup.  The driver
  replaces its `nevents` value with the requested event count.
- `Cards/param_card_active_snapshot.dat`: provenance snapshot of the live
  mutable card while the `(c3,d4)=(12,0)` point was running.  It is not the
  template for the full grid.
- `Cards/run_card_active_snapshot.dat`: live 10,000-event run-card snapshot
  captured with the active point.
- `Cards/MadLoopParams.dat`: numerical-stability and loop-reduction settings
  used by the generated process.
- `Cards/me5_configuration.txt`: MadEvent configuration from Odysseus.  Its
  absolute `mg5_path` points to the Odysseus installation and must be adjusted
  on another machine.
- `MGMEVersion.txt`: generated-process MadGraph version record.
- `SHA256SUMS`: checksums of the repository copies.

The repository copies normalize trailing whitespace and final newlines; these
changes do not alter any card setting or command.  The original source-file
hashes are recorded below so the imported copies remain traceable to the exact
Odysseus files:

| Source file | Odysseus SHA-256 |
|---|---|
| `run_c3d4_bridge_10k.sh` | `63f645685194b6720b6a0a121dd368044896a1e01478fb35ec91955ee3196d5a` |
| pre-scan `param_card.dat` | `1f27806cc459fbf807012c227377c587fe4cfa3889e746789923ef5c5159657c` |
| pre-scan `run_card.dat` | `578c0e826dc686e915937c11452a222deca953253b33d8c969ab201acd8b813b` |
| active `param_card.dat` | `e87809127171955df7cf8c25b33ca8a677e57d3de9abcf633fe96fec6cfb2679` |
| active `run_card.dat` | `aa48a5f30b289494393c215243a82521f128aa89f430fb5cc613ce06f27b657e` |
| `proc_card_mg5.dat` | `afc43483bfcea178b65027d9936217e47167314e3f7a225d8fae01c6c0c26d87` |
| `MadLoopParams.dat` | `5eb9b098b1971c582ec221150eb3079b937c17b3bd1ba58666b8c538a91c86cd` |
| `me5_configuration.txt` | `6b35da20df7b4604c820b9af0395d2474022d7d7a5f3f1ef34cc637b48919b77` |
| `MGMEVersion.txt` | `f48026f5c45fd6d33eeaade23fc9a66b4998f95fac14e910b588c9510f56b21a` |

## Point set

The driver fills the regular bridge grid

```text
c3 = {-12,-9,-6,-3,0,3,6,9,12}
d4 = {-300,-200,-100,0,100,200,300}
```

and omits the three pre-existing points `(0,-100)`, `(0,0)`, and `(0,100)`.
This leaves 60 new points.  Its first 30 points form a balanced initial
refinement across both coupling signs and the three non-zero `|d4|` layers.

## Reproduction

1. Install MadGraph5_aMC@NLO 3.5.15 and make the `loop_sm_c3d4` model
   available.  The model itself is not vendored here.
2. From the MadGraph installation, generate the process with
   `Cards/proc_card_mg5.dat`.
3. In the resulting `gg_4h_c3d4` process directory, copy
   `param_card_template.dat` to `Cards/param_card.dat`,
   `run_card_template.dat` to `Cards/run_card.dat`, and install the archived
   `MadLoopParams.dat` and machine-adjusted `me5_configuration.txt` in
   `Cards/`.
4. Copy `run_c3d4_bridge_10k.sh` to the process-directory root and make it
   executable.
5. Preview the exact point order without modifying cards:

   ```bash
   ./run_c3d4_bridge_10k.sh --events 10000 --cores 192 --dry-run
   ```

6. Start or resume the campaign:

   ```bash
   ./run_c3d4_bridge_10k.sh --events 10000 --cores 192
   ```

The driver deliberately forbids concurrent instances because every point
shares the same mutable `Cards/param_card.dat` and `Cards/run_card.dat`.
