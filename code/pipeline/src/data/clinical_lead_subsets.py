"""Prespecified fixed lead subsets for the Fixed-clinical-K Phase 2A baseline.

Defined before any Phase 2A test result exists (see docs/phase2a_design.md
section 6), so a later change to these subsets is a visible deviation from
this file's git history, not a silent post-hoc adjustment.

PTB-XL's 12-lead order, verified directly from a record's .hea file (not
assumed): I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
"""
from __future__ import annotations

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NUM_LEADS = len(LEAD_NAMES)

# Rationale for each subset is in docs/phase2a_design.md section 6:
# K=1: standard single-lead rhythm-monitoring choice (clear P-wave/QRS from
#      one channel, close to the heart's mean electrical axis).
# K=3: territorial coverage (inferior / anteroseptal / lateral).
# K=6: the complete frontal-plane (limb) hexaxial view, cheapest possible
#      6-lead-equivalent electrode configuration.
FIXED_CLINICAL_SUBSETS: dict[int, list[str]] = {
    1: ["II"],
    3: ["II", "V2", "V5"],
    6: ["I", "II", "III", "aVR", "aVL", "aVF"],
}


def lead_indices(names: list[str]) -> list[int]:
    return [LEAD_NAMES.index(n) for n in names]


def fixed_clinical_indices(k: int) -> list[int]:
    if k not in FIXED_CLINICAL_SUBSETS:
        raise ValueError(f"No fixed clinical subset defined for K={k}; only {sorted(FIXED_CLINICAL_SUBSETS)}")
    return lead_indices(FIXED_CLINICAL_SUBSETS[k])
