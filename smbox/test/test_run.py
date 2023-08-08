import pandas as pd
import openml
from sklearn.model_selection import train_test_split

def fetch_open_ml_data(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    print(dataset)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["target"] = y

    return df, 'target'

# def source_pmlb_benchmark_data(dataset, cache_loc='None'):
#    """Source benchmarking dataset
#    :param dataset. name of the dataset
#    :param cache_loc. location to cache datasets previously loaded"""
#    print('Fetching the', dataset, 'benchmark data...')
#    try:
#        df = fetch_data(dataset, local_cache_dir=cache_loc)
#    except:
#        df = fetch_data(dataset)
#
#    target_name = 'target'
#    print(f'Data loaded: {df.shape}')
#    return df, target_name

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

if __name__ == "__main__":

    import sys
    #sys.path.insert(0, "./smbox")
    sys.path.insert(0, "..")

    from Utils import Logger
    from ParamSpace import ParamSpace
    from Optimise import Optimise

    logger = Logger()

    ##---- config
    search_strategy_config_ = {'lf_init_ratio': 0.3
        , 'lf_init_n': 35
        , 'lf_ratio': 1.00
        , 'alpha_n': 2
        , 'inc_rand': 'Y'
        , 'inc_pseudo_rand': 'N'}

    global config
    config = {'dataset_source': 'openml'
        , 'dataset': 38
        , 'algorithm': 'rf'
        , 'search_strategy': 'smbox'
        , 'search_strategy_config': search_strategy_config_
        , 'wallclock': 600
        , 'output_root': '/Users/salhit/development/smbox/smbox/test/resources/output/'
              }
    logger.log(f'Experiment Config: {config}', 'DEBUG')

    _random_seed = 42

    ##----

    # Code to execute if script is run directly
    #df, target_name = fetch_open_ml_data(dataset_id=config['dataset'])
    df = pd.read_csv('test/resources/dataset_38.csv')
    target_name = 'target'
    data_all = prepare_data(df, target_name, _random_seed, test_size=0.3)

    if config['algorithm'] == 'xgb':
        cfg_schema = ParamSpace.create_cfg_schema('test/resources/xgb_schema.json')

        classes = data_all['y_train'].value_counts()
        class_0 = min(classes.index.values)
        class_1 = max(classes.index.values)
        balance_ratio = round(classes[class_0] / classes[class_1], 2)

        cfg_schema['fix']['scale_pos_weight'] = balance_ratio
        logger.log(f"Updated scale_pos_weight in cfg space --> {balance_ratio}", 'DEBUG')

        cfg_schema = ParamSpace.create_cfg_schema('test/resources/xgb_schema.json')
    elif config['algorithm'] == 'rf':
        cfg_schema = ParamSpace.create_cfg_schema('test/resources/rf_schema.json')

    logger.log(f'-------------Starting SMBOX')
    logger.log(f'Initial configuration schema: {cfg_schema}', 'DEBUG')

    optimiser = Optimise(config)
    optimiser.SMBOXOptimise(data_all, cfg_schema, _random_seed)

