import joblib
import numpy as np


def predict(features):
    model = joblib.load("models/model.pkl")
    prediction = model.predict([features])
    return prediction[0]


if __name__ == "__main__":
    test_features = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
    result = predict(test_features)
    print(f"Predicted house value: ${result * 100000:.2f}")
