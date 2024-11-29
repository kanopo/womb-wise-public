import os
import warnings
import pandas as pd
import numpy as np
import re
from typing import List, Dict

warnings.simplefilter(action="ignore", category=FutureWarning)


def remove_dot_files(files: list) -> list:
    return [file for file in files if not file.startswith(".")]


def get_dataset(path: str, data_type: str) -> pd.DataFrame:
    path_base_dir = os.path.expanduser(path)
    dirs = []
    tmp = os.listdir(path_base_dir)
    tmp = remove_dot_files(tmp)
    dirs.extend(tmp)

    dataset = pd.DataFrame(
        columns=[
            "image_name",  # string value
            "leftLip_x",  # float value
            "leftLip_y",  # float value
            "rightLip_x",  # float value
            "rightLip_y",  # float value
            "topMidInner_x",  # float value
            "topMidInner_y",  # float value
            "bottomMidInner_x",  # float value
            "bottomMidInner_y",  # float value
            "nose_x",  # float value
            "nose_y",  # float value
            "test",
            "frame",
            "label",
            # "isFetus",
        ]
    )

    for root, dirs, _ in os.walk(path_base_dir):

        for d in dirs:
            splitted = root.split("/")
            path = os.path.join(root, d)
            fetus_name = splitted[-2]
            fetus_action = splitted[-3]

            if data_type == "baseline":
                if re.search(r"baseline", d) or re.search(r"Baseline", d):
                    for f in remove_dot_files(os.listdir(path)):
                        if ".png" in f and fetus_action.lower() == data_type:
                            pass

                        elif ".csv" in f and fetus_action.lower() == data_type:
                            p = os.path.join(path, f)
                            data = pd.read_csv(p)
                            data.columns = data.iloc[0]
                            data = data.drop([0, 1])
                            data = data.iloc[:, 2:]

                            data.columns = [
                                "image_name",
                                "leftLip_x",
                                "leftLip_y",
                                "rightLip_x",
                                "rightLip_y",
                                "topMidInner_x",
                                "topMidInner_y",
                                "bottomMidInner_x",
                                "bottomMidInner_y",
                                "nose_x",
                                "nose_y",
                            ]

                            data = data.dropna()
                            data["test"] = fetus_name.split("_")[1]
                            data["frame"] = p.split("/")[-2].split("_")[-1]
                            data["label"] = data_type

                            image_name = data["image_name"].apply(
                                lambda x: x.split(".")[0].split("img")[1]
                            )

                            image_name = image_name.apply(lambda x: x.zfill(4))

                            data["image_name"] = image_name

                            data = calculate_distance(data)

                            dataset = pd.concat([dataset, data])

            elif data_type == "yawn":
                if re.search(r"yawn", d) or re.search(r"Yawn", d):
                    for f in remove_dot_files(os.listdir(path)):
                        if ".png" in f and fetus_action.lower() == data_type:
                            pass

                        elif ".csv" in f and fetus_action.lower() == data_type:
                            p = os.path.join(path, f)
                            data = pd.read_csv(p)
                            data.columns = data.iloc[0]
                            data = data.drop([0, 1])
                            data = data.iloc[:, 2:]

                            data.columns = [
                                "image_name",
                                "leftLip_x",
                                "leftLip_y",
                                "rightLip_x",
                                "rightLip_y",
                                "topMidInner_x",
                                "topMidInner_y",
                                "bottomMidInner_x",
                                "bottomMidInner_y",
                                "nose_x",
                                "nose_y",
                            ]

                            data = data.dropna()
                            data["test"] = fetus_name.split("_")[1]
                            data["frame"] = p.split("/")[-2].split("_")[-1]
                            data["label"] = data_type

                            image_name = data["image_name"].apply(
                                lambda x: x.split(".")[0].split("img")[1]
                            )

                            image_name = image_name.apply(lambda x: x.zfill(4))

                            data["image_name"] = image_name
                            data = calculate_distance(data)

                            dataset = pd.concat([dataset, data])
            elif data_type == "opcl":
                if re.search(r"opcl", d) or re.search(r"Opcl", d):
                    for f in remove_dot_files(os.listdir(path)):
                        if ".png" in f and fetus_action.lower() == data_type:
                            pass

                        elif ".csv" in f and fetus_action.lower() == data_type:
                            p = os.path.join(path, f)
                            data = pd.read_csv(p)
                            data.columns = data.iloc[0]
                            data = data.drop([0, 1])
                            data = data.iloc[:, 2:]

                            data.columns = [
                                "image_name",
                                "leftLip_x",
                                "leftLip_y",
                                "rightLip_x",
                                "rightLip_y",
                                "topMidInner_x",
                                "topMidInner_y",
                                "bottomMidInner_x",
                                "bottomMidInner_y",
                                "nose_x",
                                "nose_y",
                            ]

                            data = data.dropna()
                            data["test"] = fetus_name.split("_")[1]
                            data["frame"] = p.split("/")[-2].split("_")[-1]
                            data["label"] = data_type

                            image_name = data["image_name"].apply(
                                lambda x: x.split(".")[0].split("img")[1]
                            )

                            image_name = image_name.apply(lambda x: x.zfill(4))

                            data["image_name"] = image_name
                            data = calculate_distance(data)

                            dataset = pd.concat([dataset, data])

    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    return dataset


def euclidean_distance(x1, y1, x2, y2) -> float:
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_distance(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["top_bottom_distance"] = dataset.apply(
        lambda x: euclidean_distance(
            x["topMidInner_x"],
            x["topMidInner_y"],
            x["bottomMidInner_x"],
            x["bottomMidInner_y"],
        ),
        axis=1,
    )
    return dataset


def split_dataset(
    dataset: pd.DataFrame, classes, train_optimal_size, test_optimal_size, series_length
) -> List[Dict[np.ndarray, int]]:
    dataset["label"] = dataset["label"].apply(lambda x: classes.index(x))

    total_dataset: np.ndarray = []

    grouped = dataset.groupby(by=["fetus", "frame", "label"])
    for name, group in grouped:
        df = group.reset_index(drop=True, inplace=False)
        # remove a col
        label = df.pop("label")
        frame = df.pop("frame")
        fetus = df.pop("fetus")
        distance = df.pop("top_bottom_distance")

        df = df.sort_values(by="image_name")
        df.pop("image_name")
        df = df.reset_index(drop=True, inplace=False)
        x = df.values
        # x = distance.values
        y = label.values[0]

        if x.shape[0] < series_length:
            missing_cols = series_length - x.shape[0]
            if len(x.shape) == 1:
                new_matrix = np.full((missing_cols), x[0])
            else:
                new_matrix = np.zeros((missing_cols, x.shape[1]))
            x = np.concatenate((x, new_matrix), axis=0)

        if x.shape[0] > series_length:
            if len(x.shape) == 1:
                x = x[:series_length]
            else:
                x = x[:series_length, :]

        total_dataset.append({"x": x, "y": y})

        # if y.shape[0] < series_length:
        #     missing_cols = series_length - y.shape[0]
        #     new_matrix = np.full((missing_cols), y[0])
        #     y = np.concatenate((y, new_matrix), axis=0)
        #
        # if y.shape[0] > series_length:
        #     y = y[:series_length]

    total_yawn_count = 0
    total_baseline_count = 0
    total_opcl_count = 0
    for i in range(len(total_dataset)):
        if total_dataset[i]["y"] == 0:
            total_baseline_count += 1

        if total_dataset[i]["y"] == 1:
            total_opcl_count += 1

        if total_dataset[i]["y"] == 2:
            total_yawn_count += 1

    train: List[Dict[np.ndarray, int]] = []
    test: List[Dict[np.ndarray, int]] = []
    # val: List[Dict[np.ndarray, int]] = []

    train_yawn_optimal_size = int(total_yawn_count * train_optimal_size)
    test_yawn_optimal_size = int(total_yawn_count * test_optimal_size)

    train_baseline_optimal_size = int(total_baseline_count * train_optimal_size)
    test_baseline_optimal_size = int(total_baseline_count * test_optimal_size)

    train_opcl_optimal_size = int(total_opcl_count * train_optimal_size)
    test_opcl_optimal_size = int(total_opcl_count * test_optimal_size)

    train_yawn: List[Dict[np.ndarray, int]] = []
    test_yawn: List[Dict[np.ndarray, int]] = []
    # val_yawn: List[Dict[np.ndarray, int]] = []

    train_baseline: List[Dict[np.ndarray, int]] = []
    test_baseline: List[Dict[np.ndarray, int]] = []
    # val_baseline: List[Dict[np.ndarray, int]] = []

    train_opcl: List[Dict[np.ndarray, int]] = []
    test_opcl: List[Dict[np.ndarray, int]] = []
    # val_opcl: List[Dict[np.ndarray, int]] = []

    for data in total_dataset:
        if data["y"] == 2:
            # yawn
            if len(train_yawn) < train_yawn_optimal_size:
                train_yawn.append({"data": data["x"], "label": data["y"]})
            else:
                test_yawn.append({"data": data["x"], "label": data["y"]})
            # elif (
            #     len(val_yawn)
            #     < total_yawn_count - train_yawn_optimal_size - test_yawn_optimal_size
            # ):
            #     val_yawn.append({"data": data["x"], "label": data["y"]})
        elif data["y"] == 0:
            # baseline
            if len(train_baseline) < train_baseline_optimal_size:
                train_baseline.append({"data": data["x"], "label": data["y"]})
            else:
                test_baseline.append({"data": data["x"], "label": data["y"]})
            # elif (
            #     len(val_baseline)
            #     < total_baseline_count
            #     - train_baseline_optimal_size
            #     - test_baseline_optimal_size
            # ):
            #     val_baseline.append({"data": data["x"], "label": data["y"]})
        elif data["y"] == 1:
            # opcl
            if len(train_opcl) < train_opcl_optimal_size:
                train_opcl.append({"data": data["x"], "label": data["y"]})
            else:
                test_opcl.append({"data": data["x"], "label": data["y"]})
            # elif (
            #     len(val_opcl)
            #     < total_opcl_count - train_opcl_optimal_size - test_opcl_optimal_size
            # ):
            #     val_opcl.append({"data": data["x"], "label": data["y"]})
        else:
            print("[ERROR] Invalid class label during split")
            break

    train += train_yawn + train_baseline + train_opcl
    test += test_yawn + test_baseline + test_opcl
    # val += val_yawn + val_baseline + val_opcl

    np.random.shuffle(train)
    np.random.shuffle(test)
    # np.random.shuffle(val)

    return train, test
