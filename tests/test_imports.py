import numpy as np
import pytest
import mlflow
import xgboost
import catboost
import openml

def test_imports():
    """Test that all required packages can be imported."""
    assert np.__version__
    assert mlflow.__version__
    assert xgboost.__version__
    assert catboost.__version__
    assert openml.__version__ 