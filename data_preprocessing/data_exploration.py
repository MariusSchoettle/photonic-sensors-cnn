from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import random

assignment_path = Path(
    "./data_preprocessing/split_assignment/60-20-20-split_7200-seconds/"
)
train_path = assignment_path / "train.csv"
df_train = pd.read_csv(train_path)

figure = plt.figure()
for i in range(9):
    random_sample = df_train.iloc[random.randint(0, len(df_train))]
    img = Image.open(random_sample["filepath"])
    temperature, time = random_sample["temperature"], random_sample["time"]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"T: {temperature} °C, t: {time} s\nShape: {img.size}", fontsize=8)
plt.show()
