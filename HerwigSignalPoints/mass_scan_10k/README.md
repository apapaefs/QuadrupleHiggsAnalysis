# Extended-scalar Herwig mass scan

This campaign showers and hadronizes the completed MadGraph LHE samples for:

- `gg_iota0_hhhh`: 42 direct-resonance mass points;
- `gg_iota0_eta0eta0_hhhh`: 441 cascade mass points.

The cards are generated from the c3-d4 Herwig signal card. They retain its
Herwig 7.3.0 shower, PDF, MPI, hadronization, forced `h0 -> b bbar`, and HwSim
settings. Both BSM scalar PDG codes are declared so the resonances stored in
the LHE records can be read. HwSim uses anti-kt R=0.4 jets and
`GhostBHadrons`, including the installed `bHadronMultiplicity` branch.

All products are written below this directory. Existing MadGraph events and
c3-d4 Herwig outputs are never modified.

## Preparation

From the repository root:

```bash
python3 HerwigSignalPoints/mass_scan_10k/prepare_herwig_mass_scan.py
```

Preparation validates all mass hierarchies, requires 42 non-empty direct LHE
files and 441 non-empty cascade LHE files, assigns deterministic unique seeds,
and writes `manifest.csv` plus the three input lists.

## Dry run and production

```bash
./HerwigSignalPoints/mass_scan_10k/run_mass_scan.sh all 8 --dry-run

nohup ./HerwigSignalPoints/mass_scan_10k/run_mass_scan.sh all 8 \
  > HerwigSignalPoints/mass_scan_10k/production.log 2>&1 &
```

The second argument is the number of simultaneous Herwig instances. Each
instance is constrained to one numerical-library thread, so `8` means eight
concurrent single-process jobs.

Run only one model with `direct` or `cascade` in place of `all`. Repeating the
same command resumes the campaign: complete ROOT outputs with a completion
marker are skipped. Use `--force` only when deliberate regeneration is wanted.

## Monitoring

```bash
tail -f HerwigSignalPoints/mass_scan_10k/production.log
python3 HerwigSignalPoints/mass_scan_10k/status_mass_scan.py
python3 HerwigSignalPoints/mass_scan_10k/status_mass_scan.py --verbose
```

Per-point read/run logs and ROOT outputs are under the `direct/` and
`cascade/` subdirectories.
