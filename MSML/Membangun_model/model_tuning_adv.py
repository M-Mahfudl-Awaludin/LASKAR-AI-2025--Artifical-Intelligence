import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
)

# === Setup credentials for DagsHub (REQUIRED) ===
os.environ["MLFLOW_TRACKING_USERNAME"] = "M-Mahfudl-Awaludin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "e70767533d23a034dc49370d0e59b7e8226b039a"  # <- Ganti token kamu

# === Init DagsHub ===
from dagshub import init
init(repo_owner='M-Mahfudl-Awaludin', repo_name='Eksperimen-SML', mlflow=True)

# === Load preprocessed dataset ===
df = pd.read_csv("processed_data.csv")
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Manual tracking experiment ===
mlflow.set_experiment("RandomForest_Tuning_Advanced")

# === Model & Param Grid ===
est = RandomForestClassifier(random_state=19)
params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
    "bootstrap": [True, False],
}

# === Hyperparameter Tuning ===
search = RandomizedSearchCV(
    estimator=est,
    param_distributions=params,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    random_state=19,
    verbose=1
)
search.fit(X_train, y_train)

# === Best model & evaluation ===
best_model = search.best_estimator_
best_params = search.best_params_

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# === Start MLflow Run ===
with mlflow.start_run():
    mlflow.log_params(best_params)

    # Log metrics (manual + tambahan)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)
    mlflow.log_metric("true_positive", tp)

    # Log model to DagsHub
    mlflow.sklearn.log_model(best_model, "model", input_example=X_test.iloc[:5])

    # Log confusion matrix heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    heatmap_file = "conf_matrix_heatmap.png"
    plt.savefig(heatmap_file)
    mlflow.log_artifact(heatmap_file)

# === Print summary ===
print("Best Parameters:", best_params)
print(f"Accuracy: {acc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:\n", conf_matrix)
