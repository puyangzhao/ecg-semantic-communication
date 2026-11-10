"""Channel-use accounting for Phase 2A: payload symbols + signaling overhead.

Signaling overhead is never treated as zero (docs/phase2a_design.md
section 4) — a receiver decoding a variable, sample-specific lead subset
needs to know which K of the 12 leads were sent.
"""
from __future__ import annotations

import math

NUM_LEADS = 12


def payload_symbols(k: int, d: int = 32) -> int:
    """N_payload = K x d, identical across methods at a given K."""
    return k * d


def overhead_bits_flat(num_leads: int = NUM_LEADS) -> int:
    """One bit per lead, K-agnostic upper-bound scheme."""
    return num_leads


def overhead_bits_index(k: int, num_leads: int = NUM_LEADS) -> int:
    """Information-theoretic minimum to identify which K-subset of C(num_leads, k)
    was chosen, assuming the receiver already knows k.
    """
    if k == num_leads:
        return 0  # ceiling: nothing to signal, always all leads
    n_subsets = math.comb(num_leads, k)
    if n_subsets <= 1:
        return 0
    return math.ceil(math.log2(n_subsets))


def channel_cost_report(method: str, k: int, d: int = 32, num_leads: int = NUM_LEADS) -> dict:
    """Per docs/phase2a_design.md section 4: Fixed-clinical-K needs no signaling
    once the receiver knows the static policy; every other method (including
    Random-K, conservatively, absent an assumed shared-seed mechanism) pays
    the combinatorial index cost.
    """
    payload = payload_symbols(k, d)
    flat = overhead_bits_flat(num_leads)
    index = overhead_bits_index(k, num_leads)
    signaled_index = 0 if method == "fixed_clinical_k" else index
    return {
        "method": method,
        "k": k,
        "d": d,
        "payload_symbols": payload,
        "overhead_bits_flat": flat,
        "overhead_bits_index": index,
        "signaled_overhead_bits_index": signaled_index,
        "total_reported_cost": payload + signaled_index,
    }
