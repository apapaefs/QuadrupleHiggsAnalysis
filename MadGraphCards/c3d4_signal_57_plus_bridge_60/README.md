# 117-point c3/d4 production grid

`reference_grid_manifest.csv` is the reference manifest for the auxiliary
multi-Higgs samples generated on Tiresias.  It is the union of

- the 57 production points used by the original 10k-event `hhhh` signal grid;
- the 60 additional regular bridge points generated with
  `MadGraphCards/c3d4_bridge_extra_points_10k/run_c3d4_bridge_10k.sh`.

The union contains 117 distinct `(c3,d4)` points and 29 distinct `c3` values.
Full-loop `gg -> hhhg` generation uses all 117 points.  Direct HEFT
`gg -> hh b bbar b bbar` generation is independent of `d4` and therefore
uses the 29-point unique-`c3` projection at `d4=0`.

The `status` values in this file identify rows that are eligible for the grid
loader.  They describe the requested reference grid, not the live completion
state of any particular MG5 campaign.  Use the `monitor-mg5` commands in
`ForcedSplitting/README.md` to inspect actual production progress.
