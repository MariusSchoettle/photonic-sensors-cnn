import os
from pathlib import Path
import torch
from torchvision import transforms
import pandas as pd
import json
import matplotlib.pyplot as plt
import random

random.seed(42)
torch.manual_seed(42)


def save_results(
        model: torch.nn.Module,
        logger: dict,
        fig_loss: plt.Figure,
        fig_pred: plt.Figure,
        normalization_stats: dict,
        hyperparameters: dict,
        save_dir: Path
) -> None:
    """
    Creates a new folder in "./model_results", and saves the model as .pth file, the logger as .csv file,
    plots of the loss and predictions and saves as .png files, and saves the hyperparameters as .json file.
    """

    # Create directory for saving results
    save_dir.mkdir(exist_ok=True, parents=True)

    # Find highest currently saved version, and make new directory named next highest version
    saved_models = [int(name.split("v")[-1]) for name in os.listdir(save_dir)]
    if len(saved_models) == 0:
        this_version = 0
    else:
        this_version = max(saved_models) + 1
    model_dir = save_dir / f"model_v{this_version}"
    try:
        model_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("FileExistsError! Update folder structure.")

    # Save model
    model_path = model_dir / f"model_v{this_version}.pth"
    torch.save(obj=model.state_dict(), f=model_path)

    # Save logger
    logger_path = model_dir / f"logger_v{this_version}.csv"
    df = pd.DataFrame(logger)
    df.to_csv(logger_path)

    # Save figures
    fig_loss_path = model_dir / f"fig_loss_v{this_version}.png"
    fig_loss.savefig(fig_loss_path, dpi=600, bbox_inches="tight")

    fig_pred_path = model_dir / f"fig_pred_v{this_version}.png"
    fig_pred.savefig(fig_pred_path, dpi=600, bbox_inches="tight")

    # Save hyperparameters and stats
    config_dict = {
        "hyperparameters": hyperparameters,
        "normalization_stats": normalization_stats,
    }

    hyperparams_path = model_dir / f"config_v{this_version}.json"
    with open(hyperparams_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def denormalize_labels(temperature_norm, time_norm, stats):
    """
    Uses stats-dictionary from train dataset to denormalize temperature and time values and make interpretable.
    """
    temperature = temperature_norm * stats["std_temp"] + stats["mean_temp"]
    time = time_norm * stats["std_time"] + stats["mean_time"]
    return temperature, time


def denormalize_image(image, mean_vals, std_vals):
    """
    Takes z-score normalized images of shape [3, W, H] and normalization stats to get rgb-values between 0-1.
    Then returns the image to its original aspect ratio (model takes in square images of 224x224)
    """
    image[0] = image[0] * std_vals[0] + mean_vals[0]
    image[1] = image[1] * std_vals[1] + mean_vals[1]
    image[2] = image[2] * std_vals[2] + mean_vals[2]

    inverse_transform = transforms.Resize((80, 224))  # Return to original aspect ratio
    image = inverse_transform(image)
    return image


def visualize_subset(dataset, stats):
    """
    Visualize a random set of samples from a dataset.
    """
    rows, cols = 3, 3
    plt.figure()
    for i in range(rows * cols):
        img, labels = dataset[random.randint(0, len(dataset))]
        img = img.cpu()
        img = img.permute(1, 2, 0)

        temperature, time = denormalize_labels(labels[0], labels[1], stats)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"T: {temperature}°C, t: {time}s\nShape: {list(img.shape)}", fontsize=8
        )
    plt.show()


def plot_predictions(model, dataset, device, stats, image_mean, image_std):
    """
    Make predictions on a random set of images from a test dataset and show results.
    """

    fig = plt.figure()
    model.eval()
    rows, cols = 3, 3
    for i in range(rows * cols):
        with torch.inference_mode():
            index = random.randint(0, len(dataset))
            image, labels = dataset[index]
            image = image.to(device)
            predictions = model(image.unsqueeze(dim=0))
            T_pred, t_pred = denormalize_labels(*predictions.squeeze(), stats)
            T_label, t_label = denormalize_labels(*labels, stats)

            image = denormalize_image(image, image_mean, image_std)
            image_permute = image.permute(1, 2, 0).cpu()

            plt.subplot(rows, cols, i + 1)
            plt.imshow(image_permute)
            plt.axis("off")
            plt.title(
                f"Labels: {T_label:.1f}°C, {t_label:.0f}s\nPredictions: {T_pred:.1f}°C, {t_pred:.0f}s",
                fontsize=8,
            )
    return fig


def plot_loss_evolution(logger):
    """
    Takes in the logger updated in each training step and plots the training and cv loss vs epoch number.
    """

    fig = plt.figure()
    plt.plot(logger["train_loss"], label="Training")
    plt.plot(logger["cv_loss"], label="CV")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return fig
