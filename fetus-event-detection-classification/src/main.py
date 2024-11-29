#!/usr/bin/env python3.12

import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import argparse
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from datetime import datetime

from load_dataset import get_dataset
from model import SimpleLSTM
from training import training_loop
from validation import validation

warnings.simplefilter(action="ignore", category=FutureWarning)


def setup_model_training(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    sequence_length,
    device,
    lr,
    weight_decay,
    eps,
):
    model = SimpleLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        sequence_length=sequence_length,
        device=device,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=25
    )

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    return (model, optimizer, scheduler, criterion)


def create_loaders(
    x, y, data, under=False, over=False, classes=["base", "yawn"], batch_size=2
):
    if under is True:
        x, y = undersample(x, y, data)

    if over is True:
        x, y = oversample(x, y, data)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_OPTIMAL_SIZE, random_state=seed
    )

    train_dataset = FetusDataset(
        [{"data": x, "label": y} for x, y in zip(x_train, y_train)],
        train=True,
        classes=len(classes),
    )

    test_dataset = FetusDataset(
        [{"data": x, "label": y} for x, y in zip(x_test, y_test)],
        train=True,
        classes=len(classes),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return (train_loader, test_loader)


def get_device() -> torch.device:
    if torch.backends.mps.is_built():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def undersample(x: np.ndarray, y: np.ndarray, data):
    print("Before undersampling")
    print(Counter([d["label"] for d in data]))

    rus = RandomUnderSampler(random_state=seed, sampling_strategy="majority")
    x = np.array([d["data"] for d in data])
    y = np.array([d["label"] for d in data])
    flat_x = np.array([x.flatten() for x in x])
    flat_y = np.array(y)
    x, y = rus.fit_resample(flat_x, flat_y)
    x = x.reshape(-1, SERIES_LENGTH, FEATURE_SIZE)

    print("After undersampling")
    print(Counter(y))
    return x, y


def oversample(x: np.ndarray, y: np.ndarray, data):
    print("Before oversampling")
    print(Counter([d["label"] for d in data]))

    ros = RandomOverSampler(random_state=seed, sampling_strategy="all")
    x = np.array([d["data"] for d in data])
    y = np.array([d["label"] for d in data])
    flat_x = np.array([x.flatten() for x in x])
    flat_y = np.array(y)
    x, y = ros.fit_resample(flat_x, flat_y)
    x = x.reshape(-1, SERIES_LENGTH, FEATURE_SIZE)

    print("After oversampling")
    print(Counter(y))
    return x, y


class FetusDataset(Dataset):
    def __init__(
        self, data: List[Dict[np.ndarray, int]], train: bool = False, classes: int = 2
    ):
        self.data = data
        self.train = train
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]["data"]
        y = self.data[idx]["label"]

        # Conversione del tipo di dato
        x = x.astype(np.float32)
        y = np.eye(self.classes)[y]

        # Conversione in tensori
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int32)

        # Gestione di valori NaN o infiniti
        x = torch.nan_to_num(
            x
        )  # Sostituisce NaN con 0 e valori infiniti con numeri molto grandi o piccoli

        # Normalizzazione solo durante il training
        if self.train:
            mean = x.mean()
            std = x.std()

            # Normalizzazione condizionale (solo se std > 0)
            if std > 0:
                x = (x - mean) / std

        return x, y


def createArgParser():
    parser = argparse.ArgumentParser(description="Womb Wise")
    parser.add_argument(
        "-rd",
        "--reload-dataset",
        action="store_true",
        help="Reload the dataset",
    )

    # path to the dataset
    parser.add_argument(
        "-p",
        "--path",
        action="store",
        help="Path to the dataset",
        default="~/Documents/womb-wise/Data/",
    )

    # epoch
    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        help="Number of epochs",
        default=10,
    )

    parser.add_argument(
        "-k",
        "--kfold",
        action="store",
        help="Number of folds for kfold cross validation",
        default=1,
    )

    parser.add_argument(
        "-o",
        "--oversampling",
        action="store_true",
        help="Apply oversampling",
    )

    parser.add_argument(
        "-u",
        "--undersampling",
        action="store_true",
        help="Apply undersampling",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        default="all",
        choices=["all", "fetus", "mother", "fetus-mother", "mother-fetus"],
        help="Choose the dataset: all, fetus, mother or train with mother and test with fetus or viceversa",
    )

    args = parser.parse_args()

    print(
        f"""
    ARGS:
    \n
    reload-dataset: {args.reload_dataset}
    path: {args.path}
    epochs: {args.epochs}
    kfold: {args.kfold}
    oversampling: {args.oversampling}
    undersampling: {args.undersampling}
    dataset: {args.dataset}
    """
    )
    return args


if __name__ == "__main__":
    CLASSES = ["baseline", "opcl", "yawn"]
    FEATURE_SIZE = 10
    SERIES_LENGTH = 60
    # SINGLE_FRAME_LENGTH = FEATURE_SIZE * SERIES_LENGTH
    BATCH_SIZE = 4
    WEIGHT_DECAY = 1e-5
    LEARNING_RATE = 1e-3
    TEST_OPTIMAL_SIZE = 0.2
    HIDDEN_SIZE = 256
    DROP_OUT = 0.0
    NUM_LAYERS = 2
    EPS = 1e-7

    # TEST_NAME = "0_k1_all"
    # TEST_NAME = "1_k1_fetus"
    # TEST_NAME = "2_k1_mother"
    # TEST_NAME = "3_k1_mother_fetus"
    # TEST_NAME = "4_k1_fetus_mother"
    # TEST_NAME = "5_k5_all"
    TEST_NAME = "6_k5_fetus"
    # TEST_NAME = "7_k5_mother"

    if not os.path.exists("output/" + TEST_NAME):
        os.makedirs("output/" + TEST_NAME)

    if not os.path.exists("output/" + TEST_NAME + "/weights"):
        os.makedirs("output/" + TEST_NAME + "/weights")

    if not os.path.exists("output/" + TEST_NAME + "/confusion_matrix"):
        os.makedirs("output/" + TEST_NAME + "/confusion_matrix")

    if not os.path.exists("output/" + TEST_NAME + "/metrics"):
        os.makedirs("output/" + TEST_NAME + "/metrics")

    # fix the seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = get_device()
    args = createArgParser()

    PATH = args.path
    EPOCHS = int(args.epochs)
    K_FOLD = int(args.kfold)
    OVER_SAMPLING = args.oversampling
    UNDER_SAMPLING = args.undersampling
    EARLY_STOPPING = True
    DATASET_TYPE = args.dataset

    if os.path.exists("dataset.csv") and args.reload_dataset is False:
        dataset = pd.read_csv("dataset.csv")
        mother = pd.read_csv("mother.csv")
        fetus = pd.read_csv("fetus.csv")
    else:
        baseline_fetus = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Ultrasound_Scans/tracked_frames/",
            "baseline",
        )
        yawn_fetus = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Ultrasound_Scans/tracked_frames/",
            "yawn",
        )
        opcl_fetus = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Ultrasound_Scans/tracked_frames/",
            "opcl",
        )

        fetus = pd.concat([baseline_fetus, yawn_fetus, opcl_fetus])

        baseline_mother = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Mothers_videos/Tracked/",
            "baseline",
        )
        yawn_mother = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Mothers_videos/Tracked/",
            "yawn",
        )
        opcl_mother = get_dataset(
            "~/Documents/kanopo/womb-wise/Data/Mothers_videos/Tracked/",
            "opcl",
        )

        mother = pd.concat([baseline_mother, yawn_mother, opcl_mother])
        fetus["type"] = "fetus"
        mother["type"] = "mother"

        fetus.to_csv("fetus.csv")
        mother.to_csv("mother.csv")

        dataset = pd.concat([mother, fetus])
        dataset.to_csv("dataset.csv")



    mother = mother.drop(columns=["top_bottom_distance"])
    fetus = fetus.drop(columns=["top_bottom_distance"])
    dataset = dataset.drop(columns=["top_bottom_distance"])
    grouped_dataset = dataset.groupby(["label", "frame", "test", "type"])
    grouped_mother = mother.groupby(["label", "frame", "test"])
    grouped_fetus = fetus.groupby(["label", "frame", "test"])

    data: List[Dict[np.ndarray, int]] = []
    mother_data: List[Dict[np.ndarray, int]] = []
    fetus_data: List[Dict[np.ndarray, int]] = []

    for name, group in grouped_mother:
        label = group["label"]
        frame = group["frame"]
        test = group["test"]

        group = group.drop(columns=["test", "frame", "label", "type"])

        group.set_index("image_name", inplace=True)

        if group.columns[0] == "Unnamed: 0":
            group = group.drop(columns=["Unnamed: 0"])

        group = group.to_numpy()

        if group.shape[0] < SERIES_LENGTH:
            group = np.vstack(
                [group, np.zeros((SERIES_LENGTH - group.shape[0], FEATURE_SIZE))]
            )

        elif group.shape[0] > SERIES_LENGTH:
            group = group[:SERIES_LENGTH]

        group = group.astype(np.float32)

        label = CLASSES.index(label.iat[0])
        mother_data.append(
            {
                "data": group,
                "label": label,
            }
        )

    for name, group in grouped_fetus:
        label = group["label"]
        frame = group["frame"]
        test = group["test"]

        group = group.drop(columns=["test", "frame", "label", "type"])

        group.set_index("image_name", inplace=True)

        if group.columns[0] == "Unnamed: 0":
            group = group.drop(columns=["Unnamed: 0"])

        group = group.to_numpy()

        if group.shape[0] < SERIES_LENGTH:
            group = np.vstack(
                [group, np.zeros((SERIES_LENGTH - group.shape[0], FEATURE_SIZE))]
            )

        elif group.shape[0] > SERIES_LENGTH:
            group = group[:SERIES_LENGTH]

        group = group.astype(np.float32)

        label = CLASSES.index(label.iat[0])
        fetus_data.append(
            {
                "data": group,
                "label": label,
            }
        )

    for name, group in grouped_dataset:
        label = group["label"]
        frame = group["frame"]
        test = group["test"]
        data_type = group["type"]

        group = group.drop(columns=["test", "frame", "label", "type"])

        group.set_index("image_name", inplace=True)

        if group.columns[0] == "Unnamed: 0":
            group = group.drop(columns=["Unnamed: 0"])

        group = group.to_numpy()

        if group.shape[0] < SERIES_LENGTH:
            group = np.vstack(
                [group, np.zeros((SERIES_LENGTH - group.shape[0], FEATURE_SIZE))]
            )

        elif group.shape[0] > SERIES_LENGTH:
            group = group[:SERIES_LENGTH]

        group = group.astype(np.float32)

        label = CLASSES.index(label.iat[0])
        data.append(
            {
                "data": group,
                "label": label,
            }
        )

    if K_FOLD == 1:

        x_all = [d["data"] for d in data]
        y_all = [d["label"] for d in data]

        x_mother = [d["data"] for d in mother_data]
        y_mother = [d["label"] for d in mother_data]

        x_fetus = [d["data"] for d in fetus_data]
        y_fetus = [d["label"] for d in fetus_data]

        (train_loader_all, test_loader_all) = create_loaders(
            x_all,
            y_all,
            data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (train_loader_mother, test_loader_mother) = create_loaders(
            x_mother,
            y_mother,
            mother_data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (train_loader_fetus, test_loader_fetus) = create_loaders(
            x_fetus,
            y_fetus,
            fetus_data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (model, optimizer, scheduler, criterion) = setup_model_training(
            input_size=FEATURE_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(CLASSES),
            sequence_length=SERIES_LENGTH,
            device=device,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eps=EPS,
        )

        if DATASET_TYPE == "all":
            train_loader = train_loader_all
            test_loader = test_loader_all

        elif DATASET_TYPE == "fetus":
            train_loader = train_loader_fetus
            test_loader = test_loader_fetus

        elif DATASET_TYPE == "mother":
            train_loader = train_loader_mother
            test_loader = test_loader_mother

        elif DATASET_TYPE == "fetus-mother":
            train_loader = train_loader_fetus
            test_loader = test_loader_mother

        elif DATASET_TYPE == "mother-fetus":
            train_loader = train_loader_mother
            test_loader = test_loader_fetus

        else:
            Exception("Invalid dataset type")

        trained_model = training_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epochs=EPOCHS,
            early_stopping=EARLY_STOPPING,
            log_dir="output/" + TEST_NAME + "/metrics",
        )

        loss, conf_matrix, classification_rep = validation(
            trained_model,
            test_loader,
            criterion,
            device,
        )

        # save classification report to a file
        df = pd.DataFrame(classification_rep).transpose()
        df.to_csv("output/" + TEST_NAME + "/metrics/classification_report.csv")

        torch.save(
            trained_model.state_dict(),
            "output/" + TEST_NAME + "/weights/model.pth",
        )

        plt.figure(figsize=(19.20, 10.80))
        plt.title("Confusion Matrix")
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=["Baseline", "Opcl", "Yawn"],
            yticklabels=["Baseline", "Opcl", "Yawn"],
            cmap="viridis",
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig("output/" + TEST_NAME + "/confusion_matrix/confusion_matrix.png")

        plt.figure(figsize=(19.20, 10.80))
        plt.title("Confusion Matrix Percentage")
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(
            conf_matrix_percent,
            annot=True,
            fmt=".2f",
            xticklabels=["Baseline", "Opcl", "Yawn"],
            yticklabels=["Baseline", "Opcl", "Yawn"],
            cmap="viridis",
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig("output/" + TEST_NAME + "/confusion_matrix/confusion_matrix_percentage.png")

    else:
        x_all = [d["data"] for d in data]
        y_all = [d["label"] for d in data]

        x_mother = [d["data"] for d in mother_data]
        y_mother = [d["label"] for d in mother_data]

        x_fetus = [d["data"] for d in fetus_data]
        y_fetus = [d["label"] for d in fetus_data]

        (train_loader_all, test_loader_all) = create_loaders(
            x_all,
            y_all,
            data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (train_loader_mother, test_loader_mother) = create_loaders(
            x_mother,
            y_mother,
            mother_data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (train_loader_fetus, test_loader_fetus) = create_loaders(
            x_fetus,
            y_fetus,
            fetus_data,
            over=OVER_SAMPLING,
            under=UNDER_SAMPLING,
            classes=CLASSES,
            batch_size=BATCH_SIZE,
        )

        (model, optimizer, scheduler, criterion) = setup_model_training(
            input_size=FEATURE_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(CLASSES),
            sequence_length=SERIES_LENGTH,
            device=device,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eps=EPS,
        )

        kf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
        model_index = 0

        x = None
        y = None

        if DATASET_TYPE == "all":
            x = [d["data"] for d in data]
            y = [d["label"] for d in data]
        elif DATASET_TYPE == "fetus":
            x = [d["data"] for d in fetus_data]
            y = [d["label"] for d in fetus_data]
        elif DATASET_TYPE == "mother":
            x = [d["data"] for d in mother_data]
            y = [d["label"] for d in mother_data]

        # TODO: HOW TO HANDLE THIS CASES? for the mixed training and validation
        else:
            Exception("Invalid dataset type")

        for train_index, test_index in kf.split(X=x, y=y):
            train_data = [data[i] for i in train_index]
            test_data = [data[i] for i in test_index]

            data = train_data + test_data

            x = [d["data"] for d in train_data]
            y = [d["label"] for d in train_data]

            (train_loader, test_loader) = create_loaders(
                x,
                y,
                data,
                over=OVER_SAMPLING,
                under=UNDER_SAMPLING,
                classes=CLASSES,
                batch_size=BATCH_SIZE,
            )

            (model, optimizer, scheduler, criterion) = setup_model_training(
                input_size=FEATURE_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=len(CLASSES),
                sequence_length=SERIES_LENGTH,
                device=device,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                eps=EPS,
            )

            trained_model = training_loop(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                epochs=EPOCHS,
                early_stopping=EARLY_STOPPING,
                log_dir="output/" + TEST_NAME + "/metrics/" + f"{model_index}",
            )

            loss, conf_matrix, classification_rep = validation(
                trained_model,
                test_loader,
                criterion,
                device,
            )

            # save classification report to a file
            df = pd.DataFrame(classification_rep).transpose()
            df.to_csv(
                "output/"
                + TEST_NAME
                + "/metrics/classification_report_"
                + str(model_index)
                + ".csv"
            )

            torch.save(
                trained_model.state_dict(),
                "output/" + TEST_NAME + "/weights/model_" + str(model_index) + ".pth",
            )
            plt.figure(figsize=(19.20, 10.80))
            plt.title("Confusion Matrix")
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt=".2f",
                xticklabels=["Baseline", "Opcl", "Yawn"],
                yticklabels=["Baseline", "Opcl", "Yawn"],
                cmap="viridis",
            )

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(
                "output/"
                + TEST_NAME
                + "/confusion_matrix/confusion_matrix_"
                + str(model_index)
                + ".png"
            )
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(19.20, 10.80))
            plt.title("Confusion Matrix Percentage")
            sns.heatmap(
                conf_matrix_percent,
                annot=True,
                fmt=".2f",
                xticklabels=["Baseline", "Opcl", "Yawn"],
                yticklabels=["Baseline", "Opcl", "Yawn"],
                cmap="viridis",
            )

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(
                "output/"
                + TEST_NAME
                + "/confusion_matrix/confusion_matrix_percentage_"
                + str(model_index)
                + ".png"
            )
            model_index += 1
