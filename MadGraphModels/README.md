# MadGraph UFO models

This directory contains the source UFO models used for the multi-Higgs event
generation campaigns:

- `heft_c3d4`: HEFT model with independent Higgs trilinear (`c3`) and
  quartic (`d4`) coupling modifiers.
- `loop_sm_c3d4`: loop-induced Standard Model UFO with the same `c3` and
  `d4` parameterization for full-top-mass matrix elements.
- `loop_sm_twoscalar_eft`: loop-induced extended neutral-scalar UFO with
  `eta0` and `iota0` states and the associated effective scalar-sector
  interactions.

The directories contain only reproducible UFO source modules and restriction
cards. MadGraph-generated Python bytecode, pickle caches, editor backups,
AppleDouble files, and hidden scratch restriction cards are intentionally
omitted.

To install a model into an MG5 tree, copy the complete model directory under
`MG5_aMC_*/models/`, retaining the directory name, then import it normally in
MG5. For example:

```text
import model loop_sm_c3d4
```

These copies were taken from the production MG5 3.5.16 installation on
`tiresias.servebeer.com` on 2026-07-11.
