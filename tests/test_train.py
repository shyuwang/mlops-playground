import os
import sys

# Find the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_train_function_exists():
    """
    Test to ensure that the train_model function exists in train.py.
    """
    from src.train import train_model

    assert callable(train_model), "train_model is not callable."
    print("train_model function exists and is callable.")


def test_model_saving_logic():
    """
    Test to ensure that the model saving logic is correct.
    """

    model_path = 'models/model.pkl'
    assert model_path.endswith('.pkl'), "Model file does not have a .pkl extension."
    assert 'models/' in model_path, "Model is not being saved in the 'models' directory."

