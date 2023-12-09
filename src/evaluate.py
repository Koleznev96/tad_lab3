import json
import os

from joblib import load
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_data(filename: str, base_path: str):
    path = os.path.join(base_path, filename)
    data = pd.read_csv(path)
    y_test = data.pop("default")
    return data, y_test


def main():
    filename = "test.csv"
    x_test, y_test = load_data(filename=filename, base_path="data/prepared")
    models = ["gradient-boosting", "random-forest", "k-neighbors"]
    for model in models:
        data = {}
        model_obj = load(os.path.join("models", f"{model}.joblib"))
        predict = model_obj.predict(x_test)
        predict_proba = model_obj.predict_proba(x_test)[:, 1]

        data["precision-score"] = precision_score(y_test, predict)
        data["recall-score"] = recall_score(y_test, predict)
        data["f1-score"] = f1_score(y_test, predict)
        data["accuracy-score"] = accuracy_score(y_test, predict)
        data["roc-auc-score"] = roc_auc_score(y_test, predict_proba)
        metric_filename = f"{model}.json"
        with open(f"scores/{metric_filename}", "w") as file:
            json.dump(data, file)


if __name__ == "__main__":
    main()
