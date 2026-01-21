import os
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # Load Wine dataset
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["cultivar"] = wine.target

    # Select exactly 6 features (allowed by project)
    features = [
        "alcohol",
        "malic_acid",
        "total_phenols",
        "flavanoids",
        "color_intensity",
        "proline"
    ]

    X = df[features]
    y = df["cultivar"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling (MANDATORY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(model, "model/wine_cultivar_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    print("\nModel and scaler saved successfully in /model folder.")


if __name__ == "__main__":
    main()
