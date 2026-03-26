# Standard library
import random
from pathlib import Path
from timeit import default_timer as timer
import os

# Third party libraries
import torch
from torch.utils.data import DataLoader
import torchvision

# import modules
from data_setup import TTI_Dataset
import pretrained_model
import engine
import utils

# Set hyperparameters
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MODEL = "efficientnet_b0"
WEIGHTS = "EfficientNet_B0_Weights"
FREEZE_WEIGHTS = True
DROPOUT_PROB = 0.5
OPTIMIZER = torch.optim.Adam
WEIGHT_DECAY = 7e-4
LOSS_FN = torch.nn.MSELoss
RANDOM_SEED = 42
NUM_WORKERS = 0  # os.cpu_count()

# Data directory
DATA_FOLDER = Path(os.path.abspath(__file__)).parents[1] / "data_preprocessing/split_assignment"
ASSIGNMENT_FOLDER = DATA_FOLDER / "60-20-20-split_300-seconds"


# Deep Learning
def main():
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on device: {device}")

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Set assignment files
    train_csv = ASSIGNMENT_FOLDER / "train.csv"
    cv_csv = ASSIGNMENT_FOLDER / "cv.csv"
    test_csv = ASSIGNMENT_FOLDER / "test.csv"

    # Transform and load data into datasets
    default_transform = getattr(torchvision.models, f"{WEIGHTS}").DEFAULT.transforms()
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=default_transform.mean, std=default_transform.std
            ),
        ]
    )

    train_data = TTI_Dataset(assignment_csv=train_csv, transform=transform)
    cv_data = TTI_Dataset(
        assignment_csv=cv_csv, transform=transform, stats=train_data.stats
    )
    test_data = TTI_Dataset(
        assignment_csv=test_csv, transform=transform, stats=train_data.stats
    )

    # utils.visualize_subset(train_data, train_data.stats)

    # Instantiate dataloader objects from datasets for iteration through batches
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    cv_dataloader = DataLoader(
        dataset=cv_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Model, optimizer and loss function
    model = pretrained_model.CustomConvModel(
        input_shape=list(train_data[0][0].shape),
        pretrained_model=MODEL,
        pretrained_weights=WEIGHTS,
        freeze_weights=FREEZE_WEIGHTS,
        dropout_prob=DROPOUT_PROB)
    model.to(device)

    optimizer = OPTIMIZER(
        params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = LOSS_FN()

    # Run training loops
    logger = {"train_loss": [], "cv_loss": []}

    start = timer()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------------------")
        train_loss, cv_loss = engine.train_cv_step(
            train_dataloader=train_dataloader,
            test_dataloader=cv_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            model=model,
            device=device,
        )

        logger["train_loss"].append(train_loss)
        logger["cv_loss"].append(cv_loss)

    end = timer()
    print(f"Duration: {end - start:.0f} seconds")

    # Prepare results for saving
    fig_loss = utils.plot_loss_evolution(logger)
    fig_pred = utils.plot_predictions(
        model,
        test_data,
        device=device,
        stats=train_data.stats,
        image_mean=default_transform.mean,
        image_std=default_transform.std
    )
    hyperparameters = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model": MODEL,
        "weights": WEIGHTS,
        "freeze weights": FREEZE_WEIGHTS,
        "dropout_probability": DROPOUT_PROB,
        "optimizer": str(OPTIMIZER),
        "weight_decay": WEIGHT_DECAY,
        "loss_fn": str(LOSS_FN),
        "Assignment_folder": str(ASSIGNMENT_FOLDER),
        "num_workers": NUM_WORKERS,
    }

    # Save results
    save_path = Path(os.path.abspath(__file__)).parent / "model_results"
    utils.save_results(
        model=model,
        logger=logger,
        fig_loss=fig_loss,
        fig_pred=fig_pred,
        normalization_stats=train_data.stats,
        hyperparameters=hyperparameters,
        save_dir=save_path
    )


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method(method="spawn", force=True)
    main()
