# Procurement Request: TabPFN Fine‑Tuning Pilot

**Purpose:** Request funding to run a short pilot to fine‑tune the TabPFN model on insurance datasets and validate compute/time/cost for full experiments.

**Summary (requested):**
- Resource: 1 × A100‑80 (spot) equivalent
- Duration: 10 GPU‑hours (total)
- Provider: Vast.ai (spot marketplace)
- Pricing basis: Vast.ai A100‑80 spot estimate $1.40 / GPU‑hour
- Storage approach: local repository storage (no paid object storage required for pilot)
- Contingency: 30% of subtotal

**Evidence basis (completed local testing):**
- Source artifact: `outputs/current/tables/tabpfn_finetune_trial_results.csv`
- Measured local wall time range for representative fine-tuning trials:
	- 1000 rows, context 128, 3 steps (CPU): 28.78 s
	- 3000 rows, context 128, 1 step (CPU): 36.29 s
	- 3000 rows, context 128, 1 step (MPS): 34.92 s
- Source artifact: `outputs/current/tables/domain_finetune_study_runs.csv`
	- Total per-run wall-time median across Stage-style runs: 187.93 s

**A100-80 planning estimate (from measured local baselines):**
- Single heavier fine-tune trial (about 3000 rows, context 128, 1 step): ~9 to 18 seconds.
- Full 4-dataset Stage-B-style pass (all model arms): ~8 to 12 minutes end-to-end.
- Estimation uncertainty: ±30% until first CUDA pilot is completed.

**Cost breakdown (estimate)**
- Compute (A100‑80 spot): 10 hrs × $1.40/hr = $14.00
- Storage (local disk in repo workspace): = $0.00
- Egress / misc transfers: = $5.00

- Subtotal = $19.00
- Contingency (30%) = $5.70
- TOTAL REQUESTED (compute + storage + contingency) = $24.70 → request rounded: $25

**Justification:**
- We need a small GPU pilot to measure actual CUDA runtime and memory characteristics for fine‑tuning TabPFN on our insurance datasets, using Vast.ai as the execution platform. Local CPU/MPS testing already established reproducible baseline timing and indicates that GPU acceleration should materially reduce wall time for heavier runs, but exact speedup and VRAM headroom must be measured directly on CUDA before full-scale scheduling. This pilot will: 1) produce accurate GPU‑hours per run, 2) verify VRAM needs, 3) generate checkpoints and a reproducible runbook for scaling/HPO.

**Deliverables (from pilot)**
- Measured `time_per_sample_gpu` and extrapolated GPU‑hours for N=10k, epochs=10
- Verified instance type recommendation (VRAM check and peak memory usage)
- Checkpoint artifacts and sample outputs stored locally in the project workspace
- Short runbook with exact commands to reproduce the run
- Cost summary and recommended reservation strategy (spot vs reserved)

**Acceptance criteria**
- Pilot completes a representative small-scale run (e.g., N=1k, epochs=10) without OOM on the chosen instance, or documents mitigation steps and retries.
- Checkpoint saved locally and reload check succeeds.
- `time_per_sample_gpu` measurement recorded and documented in `docs/provisioning_gpu.md`.
- Clear recommendation for next steps (proceed with full N=10k run or request additional funding) provided after pilot review.

**Timeline**
If approved, I will schedule a short representative pilot (e.g., run `scripts/finetune_pilot.py --n_pilot 1000 --device cuda`) to measure GPU runtime and VRAM. A full-scale run (N=10k) will only be scheduled after reviewing the pilot results and receiving explicit approval.
- Approval → provision Vast.ai spot instance → run pilot → results & runbook: 0.5–2 days (depending on account setup and access).

**Risks & notes**
- Spot instances can be preempted; use checkpointing and add 20–30% buffer for retries.
- Vast.ai host quality and throughput can vary by provider; select hosts with strong uptime and verified CUDA driver compatibility.
- TabPFN weights may be gated on Hugging Face and require model term acceptance or commercial licensing; contact Prior Labs at sales@priorlabs.ai if commercial licensing is required.

**Requested by:** [Your Team / Requestor Name]
**Contact:** Scott (repo owner) — attach ticket and link to repository: [docs/provisioning_gpu.md](provisioning_gpu.md)

---
If approved, I will schedule the spot instance, run `scripts/finetune_pilot.py --n_pilot 10000 --device cuda`, collect measurements, and deliver the runbook and final cost report.
