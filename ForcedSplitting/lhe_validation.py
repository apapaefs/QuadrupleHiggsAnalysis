"""Small LHE parser used by the forced-splitting smoke tests."""

import re


class LHEParticle(object):
    def __init__(
        self,
        pid,
        status,
        mother1,
        mother2,
        color1,
        color2,
        px,
        py,
        pz,
        energy,
        mass,
    ):
        self.pid = pid
        self.status = status
        self.mother1 = mother1
        self.mother2 = mother2
        self.color1 = color1
        self.color2 = color2
        self.px = px
        self.py = py
        self.pz = pz
        self.energy = energy
        self.mass = mass


class LHEEvent(object):
    def __init__(self, header, particles):
        self.header = header
        self.particles = particles


_EVENT_RE = re.compile(r"<event>(.*?)</event>", re.DOTALL)


def _clean_event_lines(event_text):
    lines = []
    for line in event_text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return lines


def parse_particle_row(line):
    columns = line.split()
    if len(columns) < 11:
        raise ValueError("LHE particle row has fewer than 11 columns: %s" % line)
    return LHEParticle(
        pid=int(columns[0]),
        status=int(columns[1]),
        mother1=int(columns[2]),
        mother2=int(columns[3]),
        color1=int(columns[4]),
        color2=int(columns[5]),
        px=float(columns[6]),
        py=float(columns[7]),
        pz=float(columns[8]),
        energy=float(columns[9]),
        mass=float(columns[10]),
    )


def parse_lhe_events(lhe_text):
    events = []
    for match in _EVENT_RE.finditer(lhe_text):
        lines = _clean_event_lines(match.group(1))
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 1:
            raise ValueError("Malformed LHE event header")
        nup = int(header[0])
        particle_lines = lines[1 : 1 + nup]
        if len(particle_lines) != nup:
            raise ValueError("Event declares %d particles but has %d rows" % (nup, len(particle_lines)))
        events.append(LHEEvent(header=header, particles=[parse_particle_row(line) for line in particle_lines]))
    return events

