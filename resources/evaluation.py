import torch
from tqdm import tqdm


def top_k_accuracy(outputs, labels, k=5):
    _, predicted = torch.topk(outputs.logits, k, dim=1)
    correct = (predicted == labels.view(-1, 1)).sum().item()
    return correct


@torch.no_grad()
def eval(model, test_dataloader, verbose=True):
    model.eval()
    correct_top_1 = 0
    correct_top_5 = 0
    total = 0

    if verbose:
        print("Run evaluation...")
        stream = tqdm(test_dataloader)
    else:
        stream = test_dataloader

    for batch in stream:
        images = batch["image"].to(model.device)
        labels = batch["label"].to(model.device)

        outputs = model(images)
        total += labels.size(0)
        correct_top_1 += top_k_accuracy(outputs, labels, 1)
        correct_top_5 += top_k_accuracy(outputs, labels, 5)

        top_1_accuracy = correct_top_1 / total
        top_5_accuracy = correct_top_5 / total
    return {
        "top_1_accuracy": top_1_accuracy,
        "top_5_accuracy": top_5_accuracy,
    }
