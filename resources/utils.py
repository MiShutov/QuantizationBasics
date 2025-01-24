import os

import datasets
import matplotlib.pyplot as plt
from IPython.display import clear_output
from transformers import AutoImageProcessor


def prepare_dataset(dataset_name, model_name, cache_dir):
    train_dataset, test_dataset = datasets.load_dataset(
        dataset_name, cache_dir=cache_dir, split=["train[:20000]", "test[:5000]"]
    )

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    def preprocess(sample):
        if sample["image"].mode != "RGB":
            sample["image"] = sample["image"].convert("RGB")

        sample["image"] = image_processor(sample["image"], return_tensors="pt")[
            "pixel_values"
        ][0]
        return sample

    train_dataset = train_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    train_dataset.set_format(type="torch", columns=["image", "label"])
    test_dataset.set_format(type="torch", columns=["image", "label"])

    test_dataset.save_to_disk(os.path.join(cache_dir, "test"))
    train_dataset.save_to_disk(os.path.join(cache_dir, "train"))


def print_file_size(path_to_file):
    file_size = os.path.getsize(path_to_file) / 2**20
    print(f"{file_size:.03f} Mb")


def plot_loss(train_losses, learning_rates=None, figsize=(6, 3)):
    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=figsize)
    lines = []

    line_loss = ax.plot(train_losses, label="loss", color="blue")[0]
    ax.set_xlabel("time")
    ax.set_ylabel("Loss")
    lines.append(line_loss)

    if learning_rates is not None:
        ax_lr = ax.twinx()
        ax_lr.set_ylabel("Learning rate")
        line_lr = ax_lr.plot(learning_rates, label="lr", color="red")[0]
        lines.append(line_lr)

    # labels = []
    ax.legend(lines, [line.get_label() for line in lines])

    plt.title("Training Loss Over Iterations")
    plt.grid()
    plt.show()
