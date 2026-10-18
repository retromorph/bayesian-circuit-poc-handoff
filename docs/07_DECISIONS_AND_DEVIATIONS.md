# 07 — Decisions, clarifications and append-only deviations

## Initial pre-registration clarifications (2026-10-18; before any implementation)

The prior conversation provided a high-level Gate A/B/C plan. The handoff package makes the following operational choices **before data collection**:

1. **Masks:** 1 means ablated throughout, retain(C) ablates the complement inside a fixed pool. Attributions alone are not interventions.
2. **Gate A:** confirmed the four minimal circuits in A1 are cardinality 3; used 20 worlds/regime, 256 blind masks, and a 90% interval definition for the prompt-averaged causal response. Exact top-pair ranking and 16/20 recovery thresholds are fixed.
3. **Gate B:** exact alternative minimal circuits for the two redundant A and B pathways are **four size-2 retained sets**, distinct from Gate A's size-3 alternatives. The model construction must verify the desired behavior before discovery. This prevents treating a failed planted mechanism as a negative algorithm result.
4. **Gate C:** frozen 48-head candidate pool, EAP-IG screening split (64), discovery prompts (32), 128 blind test prompts and 128 blind test masks; score is mean of k={8,16,24} retained-circuit normalized discrepancies. MIB's edge-patching leaderboard is not equated to our head-zeroing experiment.
5. **Budget fairness:** masked-circuit queries that feed back into active learning count toward discovery; independent blind auditing does not. Logical model evaluations and wall time are reported separately.
6. **Competitor compatibility:** ProxySPEX returns interaction attributions; the common LightGBM surrogate and shared circuit optimizer are used for directly comparable intervention prediction and circuit selection, while official ProxySPEX is a distinct interaction-attribution diagnostic.
7. **Research decision:** a synthetic-only pass cannot produce overall GO; all A/B/C mandatory conditions are required. Scientific benchmark invalidity or missing computational resources must be labeled INCONCLUSIVE/BLOCKED, not silently PASS.
8. **Infrastructure:** local MLflow on the remote SSH host is bound to loopback and accessed using SSH port forwarding. The user's alternative public `0.0.0.0`/wildcard/CORS/UFW example is intentionally not the default because it exposes unauthenticated services.
9. **Additional validity clarity:** an informative Gate C candidate set requires mean clean-vs-all-pool-ablate logit-margin discrepancy >=0.5. This threshold is a preregistered operational sanity check; failure makes the experiment inconclusive, not negative evidence against the Bayesian method.
10. **Computational simplification:** Gate C models mean intervention effects over a fixed prompt batch; individualized context effects are a secondary analysis. This prevents inventing strong prompt-conditional inference from only 32 discovery prompts per mask.
11. **A3 false-alarm safeguard:** A3 detector success counts toward A-5 only if it has at most 12 false-positive world flags among all 60 A0–A2 worlds. This excludes trivial always-abstain strategies.
12. **Synthetic duplicates:** A0/A1/A3 replicated context responses are not independent statistical observations; collapse to one observation per mask (A2 uses one per context group).

## Deviation entry template (append, never erase existing entries)

### [UTC timestamp] — [short title]

- **Affected experiment / criterion:**
- **What changed:**
- **Reason:**
- **When relative to blind-outcome inspection:** before / after / unknown
- **Original protocol / config SHA-256:**
- **Updated protocol / config SHA-256 (if applicable):**
- **Which run IDs and data are affected:**
- **Statistical classification:** confirmatory / exploratory / invalidated / blocked
- **Impact on formal Go/No-Go verdict:**
- **Remediation / new preregistration needed:**

## Deviations during execution

_None at handoff. Implementation and experiments have not yet been executed._
