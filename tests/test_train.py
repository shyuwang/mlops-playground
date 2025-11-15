import os
import sys

# Find the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_artifact_is_created():
    """
    Test to ensure that the model artifact is created after training.
    """
    from src.train import train_model 

    train_model()

    # Check if the model artifact exists
    assert os.path.exists('models/model.pkl'), "Model artifact was not created."

    # Check if the model artifact is not empty
    file_size = os.path.getsize('models/model.pkl')
    assert file_size > 0, "Model artifact is empty."
    print("Model artifact created successfully with size:", file_size)

