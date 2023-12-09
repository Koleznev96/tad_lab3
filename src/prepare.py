import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import sigmaclip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


STATE = 42
NUMBER_COLUMNS = [
    "client_id",
    "age",
    "score_bki",
    "bki_request_cnt",
    "region_rating",
    "income",
    "sna",
    "first_time",
]
CATEGORY_COLUMNS = [
    "app_date",
    "education",
    "sex",
    "car",
    "car_type",
    "foreign_passport",
]
CATEGORY_COLUMNS_ENCODED = [
    "decline_app_cnt",
    "good_work",
    "home_address",
    "work_address",
    "default",
]

def set_type(
    data: pd.DataFrame,
    types_and_columns: dict,
) -> pd.DataFrame:
    """Задание определенного типа данных столбцам."""
    for type_column, columns in types_and_columns.items():
        for column in columns:
            data[column] = data[column].astype(type_column)
    return data


def formatting_data(
    data: pd.DataFrame,
    number_columns: list,
    category_columns: list,
    category_columns_encoded: list,
) -> pd.DataFrame:
    """Стандартизация данных и избавление от выбросов."""
    scaler = MinMaxScaler()
    encoders = pd.get_dummies(data[category_columns])
    for number_column in number_columns:
        _, low, upp = sigmaclip(data[number_column], 3, 3)
        dropped_values = data[number_column][
            (data[number_column] < low) | (data[number_column] > upp)
        ]
        data.drop(dropped_values.index)
    data[number_columns] = scaler.fit_transform(data[number_columns])
    new_data_frame = encoders.join(data[category_columns_encoded])
    new_data_frame = new_data_frame.join(data[number_columns])
    return new_data_frame


def run_prepare(filename: str, base_path: str) -> None:
    data = pd.read_csv(os.path.join(base_path, filename))
    data = set_type(data, {"category": CATEGORY_COLUMNS})
    x = formatting_data(
        data,
        NUMBER_COLUMNS,
        CATEGORY_COLUMNS,
        CATEGORY_COLUMNS_ENCODED,
    )
    y = x.pop("default")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=0.7,
        shuffle=True,
        random_state=STATE,
    )
    x_train["default"] = y_train
    x_test["default"] = y_test
    x_train.to_csv(os.path.join(base_path, "prepared/train.csv"), index=False)
    x_test.to_csv(os.path.join(base_path, "prepared/test.csv"), index=False)


def main():
    filename = "data.csv"
    run_prepare(filename=filename, base_path="data")


if __name__ == "__main__":
    main()
