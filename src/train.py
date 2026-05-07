import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Bonus 1: Tracking MLflow tu xa (DagsHub)
# MLflow se tu dong su dung bien moi truong neu co, nhung set tuong minh se chac chan hon.
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Canh bao lech lac du lieu
    dist = y_train.value_counts(normalize=True).to_dict()
    print("Label distribution in training data:")
    for label, ratio in dist.items():
        print(f"Class {label}: {ratio:.2%}")
        if ratio < 0.10:
            print(f"!!! WARNING: Class {label} is under-represented (< 10%)")

    with mlflow.start_run():
        mlflow.log_params(params)

        # Bonus 2: Thi nghiem voi nhieu thuat toan
        model_type = params.get("model_type", "random_forest")
        
        if model_type == "random_forest":
            model_params = {
                "n_estimators": params.get("n_estimators", 100),
                "max_depth": params.get("max_depth", None),
                "min_samples_split": params.get("min_samples_split", 2),
                "random_state": 42
            }
            model = RandomForestClassifier(**model_params)
        elif model_type == "gradient_boosting":
            model_params = {
                "n_estimators": params.get("n_estimators", 100),
                "max_depth": params.get("max_depth", 3),
                "random_state": 42
            }
            model = GradientBoostingClassifier(**model_params)
        elif model_type == "logistic_regression":
            model_params = {
                "C": params.get("C", 1.0),
                "max_iter": 1000,
                "random_state": 42
            }
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        # Du doan va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: Bao cao hieu suat tu dong
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds)
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write("=== Performance Report ===\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(cr)

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file outputs/metrics.json
        # Them thong tin label distribution va model_type
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "label_distribution": {str(k): v for k, v in dist.items()}
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Luu mo hinh
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc

if __name__ == "__main__":
    if os.path.exists("params.yaml"):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
    else:
        params = {"model_type": "random_forest"}
    train(params)
