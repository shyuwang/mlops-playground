import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_predict_function_exists():
    """
    Test to ensure that the predict function exists in predict.py.
    """
    from src.predict import predict

    assert callable(predict), "predict is not callable."
    print("predict function exists and is callable.")