#!/usr/bin/env python3
"""Regression checks for Comix phase-space dangling-current cleanup."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sherpa" / "COMIX" / "Phasespace" / "PS_Generator.C"
PORTABLE_PATCH = ROOT / "patches" / "sherpa-comix-dangling-current-cleanup.patch"


class ComixPhaseSpaceCleanupTests(unittest.TestCase):
    def test_dangling_current_is_removed_from_every_registry(self) -> None:
        source = SOURCE.read_text()
        cleanup = source[source.index("Current *const dead(*cit);") : source.index("delete dead;")]
        for registry in ("m_ctt", "m_tccs", "m_cmap", "m_cbmap"):
            self.assertIn(registry, cleanup)
        self.assertNotIn("cit=--m_cur[j].erase(cit);", source)
        self.assertIn("cit=m_cur[j].erase(cit);", source)

    def test_portable_patch_targets_upstream_source_layout(self) -> None:
        patch = PORTABLE_PATCH.read_text()
        self.assertIn("a/COMIX/Phasespace/PS_Generator.C", patch)
        self.assertIn("b/COMIX/Phasespace/PS_Generator.C", patch)
        for registry in ("m_ctt", "m_tccs", "m_cmap", "m_cbmap"):
            self.assertIn(registry, patch)


if __name__ == "__main__":
    unittest.main()
