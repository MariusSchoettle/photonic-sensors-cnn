"""
Scans through all directories and images files located in the ROOT_FOLDER.
The filepath of each image is attributed to the corresponding time and temperature.
Splits the samples in each temperature folder into train, CV, and test set.
Saves three separate .csv files with filepath, temperature, time columns.
"""

# imports
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# location of raw data
DATA_FOLDER = Path(os.getenv("RAW_DATA_PATH"))

# set properties of train-cv-test split
TRAIN_FRACTION, CV_FRACTION, TEST_FRACTION = 0.6, 0.2, 0.2  # should add to 1
MIN_TIME = 60 * 2  # time in seconds before which data will be discarded
MAX_TIME = 60 * 60 * 2  # time in seconds after which data will be discarded

# initialize dataframes
df_train = pd.DataFrame({"filepath": [], "temperature": [], "time": []})
df_cv = pd.DataFrame({"filepath": [], "temperature": [], "time": []})
df_test = pd.DataFrame({"filepath": [], "temperature": [], "time": []})


def read_single_measurement(experiment_path: Path, temperature: int) -> pd.DataFrame:
    filepaths, timestamps = [], []
    for file in experiment_path.iterdir():
        date, time = file.name.split("_")[-2], file.name.split("_")[-1][:-5]
        file_datetime = datetime(
            year=int(date[:4]),
            month=int(date[4:6]),
            day=int(date[6:]),
            hour=int(time[:2]),
            minute=int(time[2:4]),
            second=int(time[4:]),
        )
        filepaths.append(str(file))
        timestamps.append(file_datetime)
    temperatures = [temperature] * len(filepaths)
    times = [int((t - min(timestamps)).total_seconds()) for t in timestamps]
    df_temp = pd.DataFrame(
        {"filepath": filepaths, "temperature": temperatures, "time": times}
    )
    df_temp = df_temp[MIN_TIME <= df_temp["time"]]
    df_temp = df_temp[df_temp["time"] <= MAX_TIME]
    return df_temp


# Loop over all temperature and sample folders
if __name__ == "__main__":

    save_folder = (
        Path("./data_preprocessing/split_assignment")
        / f"{int(TRAIN_FRACTION * 100)}-{int(CV_FRACTION * 100)}-{int(TEST_FRACTION * 100)}-split_{int(MAX_TIME)}-seconds"
    )

    if save_folder.is_dir():
        pass
    else:
        save_folder.mkdir(parents=True, exist_ok=True)

    # Loop over all temperatures
    for temperature_folder in DATA_FOLDER.iterdir():
        if temperature_folder == save_folder:
            continue
        temperature = temperature_folder.name
        print(f"\nProcessing folder with T = {temperature}°C")
        sample_folders = list(temperature_folder.iterdir())

        # Loop over all samples, read, and append to appropriate dataframe
        for idx, sample in enumerate(sample_folders):
            print(sample)
            df_temp = read_single_measurement(sample, temperature)

            if idx / len(sample_folders) < TRAIN_FRACTION:
                df_train = pd.concat([df_train, df_temp])
            elif idx / len(sample_folders) < TRAIN_FRACTION + CV_FRACTION:
                df_cv = pd.concat([df_cv, df_temp])
            else:
                df_test = pd.concat([df_test, df_temp])

    for name, df in {"train": df_train, "cv": df_cv, "test": df_test}.items():
        df["time"] = df["time"].astype(int)
        df = df.sort_values(["temperature", "filepath", "time"])
        df.to_csv(f"{save_folder}/{name}.csv", index=False)
