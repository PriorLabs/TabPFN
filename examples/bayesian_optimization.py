#  Copyright (c) Prior Labs GmbH 2026.

"""Example of Bayesian optimization with TabPFN as the surrogate model.

TabPFN predicts a full bar distribution over the target, so acquisition
functions like Expected Improvement (EI) can be computed in closed form from
the predicted logits — no Gaussian assumption needed. This follows the
PFNs4BO approach (Mueller et al., ICML 2023, https://arxiv.org/abs/2305.17535,
https://github.com/automl/PFNs4BO), using the TabPFN foundation model as the
surrogate.

The loop below minimizes the Hartmann-6 function (a classic 6D BO benchmark
where random search does poorly):

1. Fit TabPFN on the points evaluated so far via
   ``fit_with_differentiable_input`` (tensors in, gradients preserved).
2. Score a batch of random candidates with EI in a single forward pass.
3. Refine the most promising candidates by gradient *ascent on EI itself* —
   ``differentiable_input=True`` lets gradients flow from the acquisition
   value back to the candidate coordinates.
4. Evaluate the objective at the best candidate and repeat.

Runs in well under a minute on CPU; faster on a CUDA GPU.
"""

import torch
from tqdm import trange

from tabpfn import TabPFNRegressor

DIM = 6
HARTMANN_OPTIMUM = -3.32237

N_INIT = 10  # random points to seed the surrogate
N_BO_STEPS = 25  # BO iterations (one objective evaluation each)
N_CANDIDATES = 512  # random candidates screened per iteration
TOP_K = 4  # candidates refined by gradient ascent on EI
N_REFINE_STEPS = 8  # gradient steps on the candidate coordinates
REFINE_LR = 0.05

HARTMANN_A = torch.tensor(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
HARTMANN_P = 1e-4 * torch.tensor(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)
HARTMANN_ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2])


def hartmann6(x: torch.Tensor) -> torch.Tensor:
    """Hartmann-6 function on [0, 1]^6; global minimum -3.32237."""
    inner = ((x.unsqueeze(-2) - HARTMANN_P) ** 2 * HARTMANN_A).sum(-1)
    return -(HARTMANN_ALPHA * torch.exp(-inner)).sum(-1)


def expected_improvement(
    reg: TabPFNRegressor, x: torch.Tensor, best_f: float
) -> torch.Tensor:
    """EI over ``best_f`` for a batch of points, differentiable w.r.t. ``x``.

    ``forward`` returns bar-distribution logits as [N_borders, N_samples];
    after transposing, ``raw_space_bardist_.ei`` integrates the improvement
    over the predicted distribution in closed form. Because the raw-space
    borders are an affine rescaling of the z-normalized ones, the logits can
    be used with the raw-space criterion directly and ``best_f`` is passed in
    the original (unnormalized) target space.
    """
    averaged_logits, _outputs, _borders = reg.forward(x, use_inference_mode=True)
    logits = averaged_logits.transpose(0, 1).float()
    return reg.raw_space_bardist_.ei(logits, best_f, maximize=True)


def propose_next_point(
    reg: TabPFNRegressor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """One acquisition round: screen random candidates, refine the best by EI ascent."""
    # We maximize -hartmann6, so the incumbent is the current maximum.
    reg.fit_with_differentiable_input(train_x, train_y)
    best_f = train_y.max().item()

    # Stage 1: screen a cheap batch of random candidates in one forward pass.
    with torch.no_grad():
        cand_x = torch.rand(N_CANDIDATES, DIM, device=device)
        ei = expected_improvement(reg, cand_x, best_f)
        top_x = cand_x[ei.topk(TOP_K).indices]

    # Stage 2: gradient ascent on EI w.r.t. the candidate coordinates.
    refine_x = top_x.clone().requires_grad_(requires_grad=True)
    optimizer = torch.optim.Adam([refine_x], lr=REFINE_LR)
    for _ in range(N_REFINE_STEPS):
        optimizer.zero_grad()
        loss = -expected_improvement(reg, refine_x, best_f).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            refine_x.clamp_(0.0, 1.0)  # stay inside the search domain

    with torch.no_grad():
        ei_refined = expected_improvement(reg, refine_x, best_f)
        return refine_x[ei_refined.argmax()].detach()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"Device: {device}")
    print(f"Hartmann-6 global minimum: {HARTMANN_OPTIMUM:.4f}\n")

    reg = TabPFNRegressor(
        n_estimators=1,
        device=device,
        random_state=0,
        inference_precision=torch.float32,
        differentiable_input=True,
    )

    # Seed with random evaluations; store -hartmann6 so that EI (which
    # maximizes) drives the minimization.
    train_x = torch.rand(N_INIT, DIM, device=device)
    train_y = -hartmann6(train_x)

    for step in trange(N_BO_STEPS, desc="BO steps"):
        next_x = propose_next_point(reg, train_x, train_y, device)
        next_y = -hartmann6(next_x.unsqueeze(0))
        train_x = torch.cat([train_x, next_x.unsqueeze(0)])
        train_y = torch.cat([train_y, next_y])
        print(
            f"  step {step + 1:2d}: evaluated f={-next_y.item():8.4f} "
            f"| best so far f={-train_y.max().item():8.4f}"
        )

    # Random-search baseline with the same total evaluation budget.
    rand_y = hartmann6(torch.rand(N_INIT + N_BO_STEPS, DIM, device=device))

    print(f"\nBest found by TabPFN-BO:     {-train_y.max().item():.4f}")
    print(f"Best found by random search: {rand_y.min().item():.4f}")
    print(f"(global minimum:             {HARTMANN_OPTIMUM:.4f})")


if __name__ == "__main__":
    main()
