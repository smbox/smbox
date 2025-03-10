import pandas as pd
import openml
import pytest
from sklearn.model_selection import train_test_split

from smbox.utils import Logger
from smbox.paramspace import ParamSpace
from smbox.paramspace import rf_default_param_space, xgb_default_param_space
from smbox.default_objectives import rf_objective, xgb_objective
from smbox.optimise import Optimise
from smbox.smbox_config import smbox_params

@pytest.fixture
def test_data():
    df = pd.read_csv('tests/resources/dataset_38.csv')
    target_name = 'target'
    return prepare_data(df, target_name, 42, test_size=0.3)

@pytest.fixture
def base_config(request):
    return {
        'dataset_source': 'openml',
        'dataset': 38,
        'search_strategy': 'smbox',
        'search_strategy_config': smbox_params,
        'wallclock': 300,
        'output_root': './tests/resources/output/',
        'experiment_name': f'test_experiment_{request.node.name}'
    }

@pytest.fixture
def rf_config(base_config):
    config = base_config.copy()
    config['algorithm'] = 'rf'
    return config

@pytest.fixture
def xgb_config(base_config):
    config = base_config.copy()
    config['algorithm'] = 'xgb'
    return config

def test_rf_optimization(test_data, rf_config):
    """Test Random Forest optimization."""
    logger = Logger()
    logger.log(f'-------------Starting SMBOX RF Optimization')
    
    optimiser = Optimise(rf_config, rf_objective, 42, mlflow_tracking=True)
    best_params, best_score = optimiser.SMBOXOptimise(test_data, rf_default_param_space)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0

def test_xgb_optimization(test_data, xgb_config):
    """Test XGBoost optimization."""
    logger = Logger()
    logger.log(f'-------------Starting SMBOX XGB Optimization')
    
    # Update scale_pos_weight for XGBoost
    classes = test_data['y_train'].value_counts()
    class_0 = min(classes.index.values)
    class_1 = max(classes.index.values)
    balance_ratio = round(classes[class_0] / classes[class_1], 2)
    
    cfg_schema = xgb_default_param_space.copy()
    cfg_schema['fix']['scale_pos_weight'] = balance_ratio
    
    optimiser = Optimise(xgb_config, xgb_objective, 42, mlflow_tracking=True)
    best_params, best_score = optimiser.SMBOXOptimise(test_data, cfg_schema)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0

def test_custom_param_space(test_data, rf_config):
    """Test optimization with custom parameter space."""
    logger = Logger()
    logger.log(f'-------------Starting SMBOX with Custom Param Space')
    
    cfg_schema = ParamSpace.create_cfg_schema('tests/resources/rf_schema.json')
    optimiser = Optimise(rf_config, rf_objective, 42, mlflow_tracking=True)
    best_params, best_score = optimiser.SMBOXOptimise(test_data, cfg_schema)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0

def fetch_open_ml_data(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    print(dataset)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["target"] = y

    return df, 'target'

def prepare_data(df: pd.DataFrame, target_name: str, random_seed: int = 42, test_size: float = 0.3):
    """
    Prepares the data for machine learning by splitting it into train and test sets, and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset.
    target_name (str): The name of the target variable column in df.
    random_seed (int, optional): The seed for the random number generator. Default is 42.
    test_size (float, optional): The proportion of the dataset to include in the test split, between 0.0 and 1.0. Default is 0.3.

    Returns:
    dict: A dictionary containing the preprocessed pandas DataFrames, with keys 'X_train', 'y_train', and optionally 'X_test', 'y_test'.

    Note:
    If test_size is 0.0, no test set is created and the returned dictionary only includes 'X_train' and 'y_train'.
    """
    y = df[target_name]
    X = df.drop(target_name, axis=1)
    X.fillna(0, inplace=True)

    if test_size == 0.0:
        print('No test set created')
        data = {"X_train": X, "y_train": y}
    else:
        # Create test and train splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    return data

