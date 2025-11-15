import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_predict_returns_number():
    """
    Test to ensure that the predict function returns a numerical value.
    """
    from src.predict import predict
    from src.train import train_model

    if not os.path.exists('models/model.pkl'):
        train_model()


    test_features = np.array([8.3252, 41.0, 6.984127, 1.0238095, 322.0, 2.5555556, 37.88, -122.23])  # Sample input features

    # Call the predict function
    prediction = predict(test_features)

    # Check if the prediction is a number
    assert isinstance(prediction, (int, float, np.number)), "Prediction is not a numerical value."
    print("Prediction returned a numerical value:", prediction)