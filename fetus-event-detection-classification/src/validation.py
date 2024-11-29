import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
import tqdm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def validation(
    model,
    val_loader,
    criterion,
    device,
):
    # move to device
    model = model.to(device)

    # metrics

    losses = []
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y = y.squeeze(1)
            y = y.float()
            y_pred = y_pred.squeeze(1)
            y_pred = y_pred.float()
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            predictions.append(torch.argmax(y_pred, dim=1))
            ground_truth.append(torch.argmax(y, dim=1))

    loss = sum(losses) / len(losses)

    gt = torch.cat(ground_truth).cpu().numpy()
    pred = torch.cat(predictions).cpu().numpy()

    replace = {0: "base", 1: "opcl", 2: "yawn"}

    conf_matrix = confusion_matrix(gt, pred)

    gt = [replace[i] for i in gt]
    pred = [replace[i] for i in pred]

    classification_rep = classification_report(
        pred, gt, zero_division=0, output_dict=True
    )

    print(classification_rep)

    return loss, conf_matrix, classification_rep
