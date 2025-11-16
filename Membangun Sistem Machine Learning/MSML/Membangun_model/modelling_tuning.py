import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

# === Load preprocessed dataset ===
df = pd.read_csv("processed_data.csv")
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Set MLflow Tracking ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Model Tuning - RandomizedSearch")

# === Model and param grid ===
est = RandomForestClassifier(random_state=19)
params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
    "bootstrap": [True, False],
}

# === Randomized Search ===
search = RandomizedSearchCV(
    estimator=est,
    param_distributions=params,
    n_iter=20,  # Number of random combinations
    cv=3,
    n_jobs=-1,
    random_state=19,
    verbose=1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_

# === Start MLflow logging ===
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(best_params)

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)
    mlflow.log_metric("true_positive", tp)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_model_tuned",
        input_example=X_test.iloc[:5]
    )

    # === Save and log confusion matrix heatmap ===
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()

    # Save image locally
    cm_filename = "conf_matrix_heatmap.png"
    plt.savefig(cm_filename)

    # Log image to MLflow
    mlflow.log_artifact(cm_filename)

# === Print output ===
print("Best Parameters:", best_params)
print(f"Accuracy: {acc}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print("Confusion Matrix:\n", conf_matrix)
