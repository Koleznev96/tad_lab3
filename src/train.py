import os

from joblib import dump
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier


def load_data(filename: str, base_path: str):
    path = os.path.join(base_path, filename)
    data = pd.read_csv(path)
    y_train = data.pop("default")
    return data, y_train


def main():
    filename = "train.csv"
    x_train, y_train = load_data(filename=filename, base_path="data/prepared")
    models = {
        "gradient-boosting": GradientBoostingClassifier(
            max_depth=4,
            n_estimators=600,
            criterion="squared_error",
        ),
        "random-forest": RandomForestClassifier(
            max_depth=10,
            n_estimators=2000,
            criterion="gini",
            bootstrap=True,
        ),
        "k-neighbors": KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            algorithm="brute",
        ),
    }
    for name, model in models.items():
        trained_model = model.fit(x_train, y_train)
        dump(trained_model, f"models/{name}.joblib")


if __name__ == "__main__":
    main()
