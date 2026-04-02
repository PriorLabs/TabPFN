# Funding request — GPU access for TabPFN fine‑tuning (ADSWP)

## Purpose
Enable true weight fine‑tuning of `TabPFN` on insurance datasets to complete reproducible experiments, validate prior regression/classification signals, and produce deliverables (checkpoints, OOF predictions, calibration reports).

## Objectives
- Fine‑tune `TabPFN` classifiers and regressors on up to 10k samples per dataset.
- Run small HPO/replicate experiments and checkpoint frequently.
- Produce reproducible artifacts: fitted checkpoints, OOF/test CSVs, evaluation tables, and a short technical report.

## Technical requirements
- Preferred GPU class: NVIDIA A100 (40/80 GB) or H100 for fastest turnarounds.
- Acceptable alternatives (cost‑sensitive): L40, A6000, RTX A5000, or consumer 4090/3090 for smaller pilot runs.
- VRAM guideline: 16 GB minimum for moderate runs; 40–80 GB recommended for larger batches or multi‑GPU.
- Software: Python 3.9+, PyTorch with CUDA, `tabpfn`, scikit‑learn, pandas, numpy.
- Data size: TabPFN workflows are best at ~10k rows for fine‑tuning; use sampling or subsampling for larger datasets.

## Budget options (estimates)
_Assumptions: marketplace/spot A100/H100 ≈ $1–$3/hr; public cloud on‑demand typically higher. Include storage/egress + 20–30% contingency._

- **Pilot (minimal)**: 50 GPU‑hours — Request **$150**
  - Compute ≈ $70 (50 hr @ $1.4/hr)
  - Storage/egress/contingency ≈ $80

- **Research (recommended)**: 300 GPU‑hours — Request **$600**
  - Compute ≈ $420 (300 hr @ $1.4/hr)
  - Storage/other ≈ $180

- **Full HPO (complete)**: 1,200 GPU‑hours — Request **$2,000**
  - Compute ≈ $1,680 (1200 hr @ $1.4/hr)
  - Storage/contingency ≈ $320

Pick an option or give a target hourly budget and I will refine numbers using a pilot measurement.

## Pricing snapshot (live examples)
- RunPod: A100 PCIe 80GB ≈ $1.39/hr; H100 ≈ $2.39–$3.07/hr.  
- Vast.ai marketplace: A100 SXM / A100 PCIE medians often <$1/hr; RTX 4090 med ≈ $0.29/hr.  
- CoreWeave / Lambda Labs: enterprise offerings (contact sales).  
- Public clouds (GCP/AWS/Azure): on‑demand flagship instances are more expensive; use spot/preemptible for discounts.

(Prices vary by region, availability, and spot vs on‑demand.)

## Cost estimation methodology
1. Run a pilot timing using `N_pilot` rows and measure elapsed seconds per epoch.  
2. Compute `time_per_sample = elapsed_seconds / N_pilot`.  
3. `GPU_hours = (time_per_sample * N_full * epochs * trials) / 3600`.  
4. `Total cost = GPU_hours * hourly_rate + storage + contingency`.

**Example:** If `time_per_sample = 0.18s`, `N_full = 10_000`, `epochs = 10` → 5 hours. At $1.50/hr → compute cost ≈ $7.50 for that run.

## Pilot script
See `scripts/pilot_timing.py` for a small CLI to measure per‑sample fine‑tuning time and compute estimated GPU‑hours.

## Procurement recommendations
- Pilot on marketplace/spot providers (Vast.ai, RunPod) to measure throughput at low cost.  
- For reproducible final runs, use reserved or on‑demand capacity from CoreWeave/AWS/GCP with committed‑use discounts.  
- Use spot/preemptible for large HPO but checkpoint frequently.  
- Include 20–30% contingency for re‑runs/debugging/storage.

## Deliverables & timeline
- Pilot (1 week): timing data, single fine‑tuned checkpoint, estimated GPU‑hours.  
- Main experiments (2–6 weeks): full fine‑tuning runs, OOF/test CSVs, calibration + performance report, `tabpfn` checkpoints.  
- Acceptance: reproducible checkpoints + CSV artifacts and a short reproducible notebook/process description.

## Next steps (choose one)
- I can save this file in the repo (done).  
- I can prepare a ready‑to‑run pilot script (done) and adapt it to your preprocessing.  
- If you approve a budget option, I will produce a one‑page sign‑off memo.

---
*Prepared by the ADSWP TabPFN team — will refine estimates after a short pilot run.*
