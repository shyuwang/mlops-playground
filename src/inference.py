import joblib
import numpy as np
import sys
import os
from datetime import datetime


def run_inference(features, model_path="models/model.pkl"):
    """
    Load the trained model and run inference on the provided features.

    Args:
        features (list or np.array): Input features for prediction.
        model_path (str): Path to the trained model file.
    Returns:
        float: Predicted value.
    """
    # Prepare features
    if isinstance(features, str):
        features = [float(x) for x in features.split(",")]

    if len(features) != 8:
        raise ValueError(f"Expected 8 features for prediction, got {len(features)}")

    features = np.array(features).reshape(1, -1)

    # Load model
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Make sure the artifact was downloaded correctly."
        )

    model_size = os.path.getsize(model_path) / 1024
    print(f"Model size: {model_size:.2f} KB")

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    print("Input Features:")
    print("-" * 40)
    for name, value in zip(feature_names, features[0]):
        print(f"  {name:12s}: {value:10.4f}")
    print("-" * 40)

    # Predict
    print("Running prediction...")
    prediction = model.predict(features)[0]

    return prediction


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\nError: Missing features argument")
        print("\nUsage:")
        print("  python src/inference.py <features> [prediction_name]")
        print("\nExample:")
        print(
            "  python src/inference.py '8.3,41,6.98,1.02,322,2.56,37.88,-122.23' 'luxury-house'"
        )
        print("\nFeatures (8 values, comma-separated):")
        print("  1. MedInc      - Median income")
        print("  2. HouseAge    - Median house age")
        print("  3. AveRooms    - Average rooms per household")
        print("  4. AveBedrms   - Average bedrooms per household")
        print("  5. Population  - Block population")
        print("  6. AveOccup    - Average household occupancy")
        print("  7. Latitude    - Latitude")
        print("  8. Longitude   - Longitude")
        sys.exit(1)

    features_str = sys.argv[1]
    pred_name = sys.argv[2] if len(sys.argv) > 2 else "prediction"

    print("\n" + "=" * 60)
    print("Inference Pipeline")
    print("=" * 60)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"Prediction Name: {pred_name}")

    try:
        prediction = run_inference(features_str)

        print("\n" + "=" * 60)
        print(f"PREDICTION RESULT: {pred_name}")
        print("=" * 60)
        print(f"Predicted Median House Value: ${prediction * 100000:,.2f}")
        print(f"Predicted Value (in 100k):    {prediction:.4f}")
        print("=" * 60)
        print("\nInference completed successfully!\n")

        return 0

    except ValueError as e:
        print(f"\nInput Error: {e}")
        print("Please provide exactly 8 comma-separated numbers\n")
        return 1

    except FileNotFoundError as e:
        print(f"\nModel Error: {e}")
        print("Please ensure the model artifact was downloaded\n")
        return 1

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
