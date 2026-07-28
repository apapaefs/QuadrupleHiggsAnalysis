# c3/d4 40k MadGraph campaign

This campaign produces a fresh 40,000-event `gg -> hhhh` LHE sample at each
of 153 unique coupling points:

- the established 117-point signal-plus-bridge grid;
- 24 boundary-cell centres selected from the 117-point no-Optuna contour;
- 12 additional points that constrain the upper and lower contour tips.

Existing 10k runs are preserved.  New runs use group `5` and deterministic,
unique seeds.  Ten isolated MadGraph process copies run concurrently at 32
cores each.  A worker publishes a run into the canonical directory only after
checking the LHE event count, embedded c3/d4 values, and positive integrated
cross section.

Canonical output:

```text
~/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_4h_c3d4/Events/
  run_gg_4h_5_<c3>_<d4>/unweighted_events.lhe.gz
```

Monitor from the repository root:

```bash
python3 MadGraphCards/c3d4_refinement_40k/run_c3d4_40k_campaign.py status
watch -n 30 'python3 MadGraphCards/c3d4_refinement_40k/run_c3d4_40k_campaign.py status'
tail -f MadGraphCards/c3d4_refinement_40k/controller.log
```

If you attach to the screen session, detach with **Ctrl-a**, then **d**.  Do
not use Ctrl-C: it sends SIGINT to the controller and all active MadGraph
workers.  Ctrl-C is safe when leaving `tail -f` from a separate shell.

Restart the controller after an ordinary stop with the same command:

```bash
screen -dmS gg4h_c3d4_40k_153 bash -lc '
  cd ~/Projects/QuadrupleHiggsAnalysis &&
  exec python3 MadGraphCards/c3d4_refinement_40k/run_c3d4_40k_campaign.py run \
    --workers 10 --cores 32 >> MadGraphCards/c3d4_refinement_40k/controller.log 2>&1
'
```

Validated canonical LHEs are skipped.  An incomplete worker or canonical run
directory is not deleted or overwritten automatically; inspect and archive it
before restarting.

If the controller was interrupted while MadGraph was still in its survey
stage and the worker `Events/` directories are empty, the same restart command
is sufficient: the campaign rescan resets those points to pending and reruns
them with the same deterministic seeds.

After completion, perform a full validation pass with:

```bash
python3 MadGraphCards/c3d4_refinement_40k/run_c3d4_40k_campaign.py verify \
  --workers 10 --cores 32
```
