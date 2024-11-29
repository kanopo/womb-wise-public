import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
import tqdm
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import classification_report


class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def training_loop(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    early_stopping=False,
    log_dir=None,
):

    if log_dir is not None:
        logs: pd.DataFrame = pd.DataFrame(
            columns=[
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
                "train_precision",
                "test_precision",
                "train_recall",
                "test_recall",
                "train_f1",
                "test_f1",
            ]
        )

    # move to device
    model = model.to(device)

    # metrics
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    precision = Precision(task="multiclass", num_classes=3).to(device)
    recall = Recall(task="multiclass", num_classes=3).to(device)
    f1 = F1Score(task="multiclass", num_classes=3).to(device)

    t = tqdm(range(epochs))

    early_stopping = EarlyStopping(tolerance=50, min_delta=0)
    for epochs in t:

        model.train()
        train_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            y = y.squeeze(1)
            y = y.float()
            y_pred = y_pred.squeeze(1)
            y_pred = y_pred.float()
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.argmax(y, dim=1)

            train_accuracy += accuracy(y_pred, y)
            train_precision += precision(y_pred, y)
            train_recall += recall(y_pred, y)
            train_f1 += f1(y_pred, y)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_f1 /= len(train_loader)

        model.eval()
        test_loss = 0
        test_accuracy = 0
        test_precision = 0
        test_recall = 0
        test_f1 = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            y = y.squeeze(1)
            y = y.float()
            y_pred = y_pred.squeeze(1)
            y_pred = y_pred.float()
            loss = criterion(y_pred, y)
            test_loss += loss.item()

            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.argmax(y, dim=1)

            test_accuracy += accuracy(y_pred, y)
            test_precision += precision(y_pred, y)
            test_recall += recall(y_pred, y)
            test_f1 += f1(y_pred, y)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        test_precision /= len(test_loader)
        test_recall /= len(test_loader)
        test_f1 /= len(test_loader)

        if log_dir is not None:
            new_line = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy.item(),
                "test_accuracy": test_accuracy.item(),
                "train_precision": train_precision.item(),
                "test_precision": test_precision.item(),
                "train_recall": train_recall.item(),
                "test_recall": test_recall.item(),
                "train_f1": train_f1.item(),
                "test_f1": test_f1.item(),
            }

            logs = pd.concat([logs, pd.DataFrame([new_line])])

        scheduler.step(train_loss)
        t.set_description(
            f"Epoch: {epochs + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}"
        )

        early_stopping(train_loss, test_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logs.to_csv(log_dir + "/logs.csv", index=False)

    return model
