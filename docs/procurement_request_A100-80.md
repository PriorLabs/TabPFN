# Procurement Request: TabPFN Fine‑Tuning Pilot

**Purpose:** Request funding to run a short pilot to fine‑tune the TabPFN model on insurance datasets and validate compute/time/cost for full experiments.

**Summary (requested):**
- Resource: 1 × A100‑80 (spot) equivalent
- Duration: 10 GPU‑hours (total)
- Pricing basis: spot A100‑80 estimate $1.40 / GPU‑hour
- Contingency: 30% of subtotal

**Cost breakdown (estimate)**
- Compute (A100‑80 spot): 10 hrs × $1.40/hr = $14.00
- Storage (S3/GCS for checkpoints and artifacts): = $10.00 (one‑time buffer)
- Egress / misc transfers: = $5.00

- Subtotal = $29.00
- Contingency (30%) = $8.70
- TOTAL REQUESTED (compute + storage + contingency) = $37.70 → request rounded: $40

**Justification:**
- We need a small GPU pilot to measure actual GPU runtime and memory characteristics for fine‑tuning TabPFN on our insurance datasets. The CPU pilot shows ~0.01875 s/sample (N=10k, epochs=10 → ~0.52 CPU‑hr), but GPU speedup must be measured to finalize procurement or reserve capacity. This pilot will: 1) produce accurate GPU‑hours per run, 2) verify VRAM needs, 3) generate checkpoints and a reproducible runbook for scaling/HPO.

**Deliverables (from pilot)**
- Measured `time_per_sample_gpu` and extrapolated GPU‑hours for N=10k, epochs=10
- Verified instance type recommendation (VRAM check and peak memory usage)
- Checkpoint artifacts and small sample outputs stored to object storage
- Short runbook with exact commands to reproduce the run
- Cost summary and recommended reservation strategy (spot vs reserved)

**Acceptance criteria**
**Acceptance criteria**
- Pilot completes a representative small-scale run (e.g., N=1k, epochs=10) without OOM on the chosen instance, or documents mitigation steps and retries.
- Checkpoint uploaded to object store and retrievable.
- `time_per_sample_gpu` measurement recorded and documented in `docs/provisioning_gpu.md`.
- Clear recommendation for next steps (proceed with full N=10k run or request additional funding) provided after pilot review.

**Timeline**
If approved, I will schedule a short representative pilot (e.g., run `scripts/finetune_pilot.py --n_pilot 1000 --device cuda`) to measure GPU runtime and VRAM. A full-scale run (N=10k) will only be scheduled after reviewing the pilot results and receiving explicit approval.
- Approval → provision spot instance → run pilot → results & runbook: 0.5–2 days (depending on account setup and access).

**Risks & notes**
- Spot instances can be preempted; use checkpointing and add 20–30% buffer for retries.
- TabPFN weights may be gated on Hugging Face and require model term acceptance or commercial licensing; contact Prior Labs at sales@priorlabs.ai if commercial licensing is required.

**Requested by:** [Your Team / Requestor Name]
**Contact:** Scott (repo owner) — attach ticket and link to repository: [docs/provisioning_gpu.md](provisioning_gpu.md)

---
If approved, I will schedule the spot instance, run `scripts/finetune_pilot.py --n_pilot 10000 --device cuda`, collect measurements, and deliver the runbook and final cost report.
