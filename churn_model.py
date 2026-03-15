import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


def load_data(path="customer_churn_data.csv"):
    return pd.read_csv(path)


def prepare_data(df):
    df_model = df.copy()

    categorical_cols = ["contract_type", "region"]
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

    X = df_model.drop(columns=["customer_id", "churn", "churn_probability_true"])
    y = df_model["churn"]

    return X, y, df_model


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_pred, y_prob


def plot_feature_importance(model, X, output_path="feature_importance.png"):
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_churn_probability(y_prob, output_path="churn_probability_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=30)
    plt.title("Predicted Churn Probability Distribution")
    plt.xlabel("Predicted Churn Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_high_risk_customers(df_original, test_index, y_prob, threshold=0.7):
    high_risk = df_original.loc[test_index].copy()
    high_risk["predicted_churn_probability"] = y_prob
    high_risk = high_risk[high_risk["predicted_churn_probability"] >= threshold]
    high_risk = high_risk.sort_values("predicted_churn_probability", ascending=False)

    high_risk.to_csv("high_risk_customers.csv", index=False)
    return high_risk


def save_metrics(metrics, cm, output_path="model_metrics.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("CUSTOMER CHURN MODEL METRICS\n")
        f.write("=" * 40 + "\n\n")

        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nCONFUSION MATRIX\n")
        f.write(str(cm))


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Preparing data...")
    X, y, df_model = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics, cm, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    print("\nModel Performance")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nSaving outputs...")
    plot_feature_importance(model, X_train)
    plot_churn_probability(y_prob)
    high_risk = save_high_risk_customers(df, X_test.index, y_prob, threshold=0.7)
    save_metrics(metrics, cm)

    print(f"\nHigh-risk customers saved: {len(high_risk)}")
    print("Files created:")
    print("- feature_importance.png")
    print("- churn_probability_distribution.png")
    print("- high_risk_customers.csv")
    print("- model_metrics.txt")