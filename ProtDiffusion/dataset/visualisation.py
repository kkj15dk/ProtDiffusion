import matplotlib.pyplot as plt
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

def plot_dataset(dataset, bins = None):
    # Create a dictionary to hold datasets divided by label
    divided_dataset = {}

    # Iterate over the dataset
    for data in tqdm(dataset):
        cl = data["label"][0]
        length = np.mean(data["length"])
        # If the label is not in the dictionary, add it with an empty list
        if cl not in divided_dataset:
            divided_dataset[cl] = []
        # Append the sequence to the appropriate list
        divided_dataset[cl].append(length)

    # Iterate over the divided dataset
    for label, data in divided_dataset.items():
        mean = sum(data) / len(data)
        plt.hist(data, bins=bins, label=str(label), alpha=1)
        plt.legend()
        plt.xlabel("seq length",fontsize=14)
        plt.ylabel("count",fontsize=14)
        # plt.xlim(0, 2000)
        # plt.ylim(0, 32)
        plt.savefig(f'train_hist_50grouped_{label}.png')
        plt.title(f"Distribution for label {label} with mean {mean:.2f} and std {np.std(data):.2f}")
        plt.savefig(f"train_hist_50grouped_{label}.png")
        plt.clf()

dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50_grouped')
plot_dataset(dataset, bins=100)