#!/usr/bin/env python3
"""Create the synthetic gg -> hhgg LHE smoke-test fixture."""

import argparse
import math
from pathlib import Path


def _particle_row(pid, status, mother1, mother2, color1, color2, px, py, pz, energy, mass):
    return (
        "%8d %2d %4d %4d %4d %4d "
        "% .12e % .12e % .12e % .12e % .12e 0.0000e+00 9.0000e+00"
        % (pid, status, mother1, mother2, color1, color2, px, py, pz, energy, mass)
    )


def _event_text(higgs_pz, colour_start):
    beam_energy = 500.0
    higgs_mass = 125.0
    higgs_energy = math.sqrt(higgs_mass * higgs_mass + higgs_pz * higgs_pz)
    gluon_energy = beam_energy - higgs_energy
    c1 = colour_start
    c2 = colour_start + 1
    c3 = colour_start + 2
    c4 = colour_start + 3

    rows = [
        _particle_row(21, -1, 0, 0, c1, c2, 0.0, 0.0, beam_energy, beam_energy, 0.0),
        _particle_row(21, -1, 0, 0, c3, c4, 0.0, 0.0, -beam_energy, beam_energy, 0.0),
        _particle_row(25, 1, 1, 2, 0, 0, 0.0, 0.0, higgs_pz, higgs_energy, higgs_mass),
        _particle_row(25, 1, 1, 2, 0, 0, 0.0, 0.0, -higgs_pz, higgs_energy, higgs_mass),
        _particle_row(21, 1, 1, 2, c1, c2, gluon_energy, 0.0, 0.0, gluon_energy, 0.0),
        _particle_row(21, 1, 1, 2, c3, c4, -gluon_energy, 0.0, 0.0, gluon_energy, 0.0),
    ]
    return "<event>\n6 1 1.000000e+00 1.250000e+02 7.546771e-03 1.180000e-01\n%s\n</event>" % (
        "\n".join(rows)
    )


def toy_hhgg_lhe_text(events=2):
    if events < 1:
        raise ValueError("events must be positive")

    higgs_pz_values = [100.0, 80.0, 60.0, 40.0]
    event_blocks = []
    for index in range(events):
        higgs_pz = higgs_pz_values[index % len(higgs_pz_values)]
        event_blocks.append(_event_text(higgs_pz, 501 + 10 * index))

    return """<LesHouchesEvents version="3.0">
<header>
<!-- toy_hhgg SYNTHETIC fixture: software smoke test only, not for physics or rate summaries. -->
</header>
<init>
 2212 2212 7.000000e+03 7.000000e+03 0 0 230000 230000 3 1
 1.000000e+00 0.000000e+00 1.000000e+00 1
</init>
%s
</LesHouchesEvents>
""" % (
        "\n".join(event_blocks)
    )


def write_toy_hhgg_lhe(path, events=2):
    target = Path(path)
    target.write_text(toy_hhgg_lhe_text(events=events))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--events", type=int, default=2)
    args = parser.parse_args(argv)
    write_toy_hhgg_lhe(args.output, events=args.events)


if __name__ == "__main__":
    main()

