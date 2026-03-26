import torch
from tqdm import tqdm


def train_step(
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        device: str,
) -> float:
    """
    Performs a single training step through all batches of the train dataloader. Updates model parameters.
    Returns the mean training loss of the epoch.
    """

    model.train()
    running_loss = 0
    total_samples = 0

    for X, y in tqdm(train_dataloader, desc="Train step: "):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        batch_size = len(y)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / total_samples


def test_step(
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        device: str,
) -> float:
    """
    Performs a single test step through all batches of the test dataloader.
    Returns the mean test loss of the epoch.
    """
    model.eval()
    running_loss = 0
    total_samples = 0

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Test step: "):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            batch_size = len(y)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
    return running_loss / total_samples


def train_cv_step(
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        device: str,
) -> tuple:
    """
    Performs first a train step, then a CV step. Prints out and returns current train and CV loss.
    """

    train_loss = train_step(
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model=model,
        device=device,
    )

    cv_loss = test_step(
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        model=model,
        device=device,
    )

    print(f"Training loss: {train_loss:.4f}  |  CV loss: {cv_loss:.4f}")
    return train_loss, cv_loss
