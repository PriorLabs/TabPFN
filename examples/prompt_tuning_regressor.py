#  Copyright (c) Prior Labs GmbH 2026.

"""Example of prompt-tuning a TabPFN regressor.

Prompt tuning keeps the TabPFN weights frozen and instead optimizes the
in-context examples themselves: a small set of (X, y) "prompt" samples is
treated as differentiable parameters and updated by gradient descent so that
predictions on held-out queries improve.

This relies on ``differentiable_input=True``, which lets gradients flow from a
downstream loss back through ``fit_with_differentiable_input`` into the prompt
tensors. Unlike the classifier, the regressor's ``forward`` returns bar
distribution logits, so we decode a differentiable point prediction as the
expected value over the bin centers before computing an MSE loss. The example
runs on CPU but is considerably faster on a CUDA GPU.
"""

import torch
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tabpfn import TabPFNRegressor


def predict_mean(reg: TabPFNRegressor, X: torch.Tensor) -> torch.Tensor:
    """Decode a differentiable point prediction from the regressor.

    ``forward`` returns logits over the bar distribution rather than a single
    value, so we take the expected value over the bin centers (softmax-weighted)
    and undo the z-normalisation applied during fit to return raw-space
    predictions. Gradients flow through this whole computation.
    """
    averaged_logits, _outputs, borders = reg.forward(X, use_inference_mode=True)
    # averaged_logits is [N_borders, N_samples] after the transpose inside
    # forward(); bring it back to [N_samples, N_borders].
    per_sample_logits = averaged_logits.transpose(0, 1)
    border_t = torch.as_tensor(
        borders[0],
        device=per_sample_logits.device,
        dtype=per_sample_logits.dtype,
    )
    n_logits = per_sample_logits.shape[-1]
    if border_t.numel() == n_logits + 1:
        bin_centers = (border_t[:-1] + border_t[1:]) / 2.0
    else:
        bin_centers = border_t
    probs = torch.softmax(per_sample_logits.float(), dim=-1)
    pred_z = (probs * bin_centers).sum(dim=-1)
    return pred_z * float(reg.y_train_std_) + float(reg.y_train_mean_)


def eval_test(
    reg: TabPFNRegressor,
    prompt_x: torch.Tensor,
    prompt_y: torch.Tensor,
    my_dl_test: DataLoader,
    lossfn: torch.nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the current prompt on the held-out test set."""
    with torch.no_grad():
        loss_sum = 0.0
        n_items = 0
        all_true = []
        all_pred = []
        reg.fit_with_differentiable_input(prompt_x, prompt_y.flatten())
        for data_batch in my_dl_test:
            X_tests, y_tests = data_batch
            predictions = predict_mean(reg, X_tests)
            loss_sum += lossfn(predictions, y_tests.to(device)).item() * y_tests.numel()
            n_items += y_tests.numel()
            all_true.append(y_tests.cpu())
            all_pred.append(predictions.cpu())

        res_mse = loss_sum / n_items
        res_r2 = r2_score(
            torch.cat(all_true).numpy(),
            torch.cat(all_pred).numpy(),
        )
        return res_mse, res_r2


def main() -> None:
    n_prompt_samples = 200  # samples used for prompt tuning
    n_total_samples = 2000  # total samples
    n_features = 20
    do_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Prompt samples: {n_prompt_samples}")
    print(f"Total samples: {n_total_samples}")

    torch.manual_seed(42)

    # Create synthetic dataset
    X, y = make_regression(
        n_samples=n_total_samples,
        n_features=n_features,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reg = TabPFNRegressor(
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=1,
        random_state=2,
        inference_precision=torch.float32,
        differentiable_input=True,
    )

    # The prompt: the in-context examples we will optimize.
    prompt_x_tensor = torch.tensor(X_train[:n_prompt_samples], dtype=torch.float).to(
        device
    )
    prompt_y_tensor = torch.tensor(y_train[:n_prompt_samples], dtype=torch.float).to(
        device
    )

    # The query examples used to drive the prompt-tuning loss.
    train_x_tensor = torch.tensor(X_train[n_prompt_samples:], dtype=torch.float).to(
        device
    )
    train_y_tensor = torch.tensor(y_train[n_prompt_samples:], dtype=torch.float).to(
        device
    )

    test_x_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
    test_y_tensor = torch.tensor(y_test, dtype=torch.float).to(device)

    dataset_train = TensorDataset(train_x_tensor, train_y_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataset_test = TensorDataset(test_x_tensor, test_y_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    lossfn = torch.nn.MSELoss()

    # Compute initial metrics
    mse_test, r2_test = eval_test(
        reg,
        prompt_x_tensor,
        prompt_y_tensor,
        dataloader_test,
        lossfn,
        device,
    )
    print(f"Initial Test MSE={mse_test:.4f}, R²={r2_test:.4f}")

    # Mark the prompt tensors as the parameters to optimize.
    prompt_x_tensor.requires_grad_(requires_grad=True)
    prompt_y_tensor.requires_grad_(requires_grad=True)

    optim_impl = Adam([prompt_x_tensor, prompt_y_tensor], lr=1e-2)

    for epoch in range(do_epochs):
        total_loss = 0.0
        for data_batch in tqdm(
            dataloader_train, desc=f"Epoch {epoch + 1}", leave=False
        ):
            input_x_batch, input_y_batch = data_batch
            optim_impl.zero_grad()
            reg.fit_with_differentiable_input(
                prompt_x_tensor, prompt_y_tensor.flatten()
            )
            # use_inference_mode=True for correct output format; gradients still
            # flow because differentiable_input=True.
            predictions = predict_mean(reg, input_x_batch)
            loss = lossfn(predictions, input_y_batch.to(device))
            loss.backward()
            optim_impl.step()
            total_loss += loss.item()

        mse_test, r2_test = eval_test(
            reg,
            prompt_x_tensor,
            prompt_y_tensor,
            dataloader_test,
            lossfn,
            device,
        )
        print(
            f"Epoch {epoch + 1}/{do_epochs}: Test MSE={mse_test:.4f}, R²={r2_test:.4f}"
        )


if __name__ == "__main__":
    main()
