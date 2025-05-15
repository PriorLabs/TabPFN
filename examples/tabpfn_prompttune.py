"""An example of how prompt-tuning can be applied with TabPFN."""
import json
from functools import partial
from pathlib import Path

import sklearn
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tabpfn import TabPFNClassifier


def eval_test(clf, prompt_x, prompt_y, my_dl_test, lossfn):
    """Eval for prompt tuning."""
    with torch.no_grad():
        loss_sum = 0.0
        acc_sum = 0.0
        acc_items = 0
        clf.fit(prompt_x.detach(), prompt_y.flatten().detach())
        for data_batch in my_dl_test:
            X_tests, y_tests = data_batch
            predictions = clf.predict_proba_tensor(X_tests)
            loss_sum += lossfn(torch.log(predictions), y_tests.to(device)).item()
            acc_sum += accuracy_score(y_tests.flatten().cpu(), predictions[:, 1].flatten().cpu()>0.5)*y_tests.numel()
            acc_items += y_tests.numel()

        res_accuracy = acc_sum/acc_items
        return loss_sum, res_accuracy

if __name__ == "__main__":
    n_prompt_samples = 200 # samples used for prompt_tuning
    n_total_samples = 100_000 # prompt+train+test
    do_epochs = 3
    device = "cuda:1"

    # Load Covertype Dataset (7-way classification)
    data_frame_x, data_frame_y = sklearn.datasets.fetch_covtype(return_X_y=True, shuffle=True)
    ## Manual preprocessing to numerate classes from [0...6] instead of [1...7]
    data_frame_y -= 1

    splitfn = partial(train_test_split, test_size=0.3)
    X_train, X_test, y_train, y_test = splitfn(data_frame_x[:n_total_samples], data_frame_y[:n_total_samples])
    clf = TabPFNClassifier(ignore_pretraining_limits=True, device=device, n_estimators=1,
                           random_state=2, inference_precision=torch.float32, differentiable_input=True)
    # Usually this attribute will be set when fit is called the first time.
    # When using prompt tuning, the classifier is not reinitialized in subsequent fit
    # calls. This may lead to problems, if the subset used does not contain all the right
    # classes (in this dataset, class 7 is very rare). This can be circumvented by explicitly
    # setting the number of classes.
    clf.n_classes_ = 7
    prompt_x_tensor = torch.tensor(X_train[:n_prompt_samples], dtype=torch.float).to(device)
    prompt_y_tensor = torch.tensor(y_train[:n_prompt_samples], dtype=torch.float).to(device)

    train_x_tensor = torch.tensor(X_train[n_prompt_samples:], dtype=torch.float).to(device)
    train_y_tensor = torch.tensor(y_train[n_prompt_samples:], dtype=torch.long).to(device)

    test_x_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    dataset_train = TensorDataset(train_x_tensor, train_y_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataset_test = TensorDataset(test_x_tensor, test_y_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    lossfn = torch.nn.NLLLoss()

    ## Compute initial accuracy
    loss_test, res_acc = eval_test(clf, prompt_x_tensor, prompt_y_tensor, dataloader_test, lossfn)
    print("Initial accuracy:", res_acc)

    prompt_x_tensor.requires_grad_(True)  # noqa: FBT003
    prompt_y_tensor.requires_grad_(True)  # noqa: FBT003

    optim_impl = Adam([prompt_x_tensor, prompt_y_tensor], lr=5e-3)
    loss_batches = []
    acc_batches = []
    for epoch in range(do_epochs):
        for data_batch in tqdm(dataloader_train):
            input_x_batch, input_y_batch = data_batch
            optim_impl.zero_grad()
            clf.fit(prompt_x_tensor, prompt_y_tensor.flatten())
            predictions = clf.predict_proba_tensor(input_x_batch)
            loss = lossfn(torch.log(predictions), input_y_batch.to(device))
            loss.backward()
            optim_impl.step()
        loss_test, res_acc = eval_test(clf, prompt_x_tensor, prompt_y_tensor, dataloader_test, lossfn)
        print(f"---- EPOCH {epoch}: ----")
        print("Test Acc:", res_acc)
        print("Test Loss:", loss_test)
        with Path("prompttune.json").open(mode="w") as f:
            json.dump({"loss": loss_batches, "acc": acc_batches}, f)