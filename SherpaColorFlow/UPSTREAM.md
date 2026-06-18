# Upstream Sherpa Provenance

- Upstream repository: `https://gitlab.com/sherpa-team/sherpa.git`
- Base commit: `a7ba2c8b98da1fbc5e9fc290f1b8e6584afa71fe`
- Source patch: `patches/sherpa-lhef-color-flow-hack.patch`
- Original validated source path: `physres1.kennesaw.edu:~/Projects/4H/sherpa-src`
- Original validated patch path:
  `physres1.kennesaw.edu:~/Projects/4H/patches/sherpa-lhef-color-flow-hack.patch`

## Runtime Settings Added by the Patch

The colour-flow export is opt-in:

```yaml
COLOR_SCHEME: SAMPLE

SP:
  SET_COLORS: true
  LHE_COLOR_FLOW_HACK: true
  LHE_COLOR_FLOW_DEBUG: false
```

For colourless on-shell `Z -> b bbar` decay products that are not part of the
Comix sampled colour point, the LHE writer can assign one final-state singlet:

```yaml
LHEF_ASSIGN_MISSING_QQBAR_SINGLET: true
```

Cards with repeated final-state flavours should also keep final-state flavour
ordering stable:

```yaml
PROCESSES:
- 21 21 -> 5 -5 5 -5 5 -5 5 -5:
    Sort_Flavors: 1
```

## Validation Status

The patch was validated on physres1 with a 100-event
`p p -> Z + 6b`, `Z -> b bbar` LHE run. The validator checked:

- exactly two incoming partons;
- eight final-state b or bbar partons;
- no stable final-state Z row;
- mass consistency between LHE mass and four-vector;
- nonzero colour tags on all coloured rows;
- distinct nonzero tags on gluons;
- every nonzero colour tag appearing exactly twice;
- one colour endpoint paired with one anticolour endpoint;
- no self-connected colour tags;
- an isolated first final-state `b bbar` singlet for the Z decay.
