# smbox

## 📖 Introduction

Sequential Model-Based Optimization eXpress `smbox` is a powerful library for Sequential Model-Based Optimization (SMBO), which is a process for finding the optimal value of an expensive and noisy black-box function. With `smbox`, you can efficiently optimize the hyperparameters of any ML algorithm to get the best performance without an exhaustive search.

- 🚂 **State-of-the-art HPO**: Achieve state-of-the-art hyperparameter optimization (HPO) performance with minimal effort.


- 🎯 **Default Parameter Space**: Use our default parameter search space to simply plug and play with the most common ML algorithms.
Making it simpler than defining grid search ranges or complicated parameter search distributions.


- 🤖️ **Bring your own Parameter Space**: Provide your own parameter search space to better suit your needs, or update our default search spaces with ease.


- 🤖️ **Bring your own Objective Function**: Tune any ML algorithm using the performance metric to suit your needs with. (Coming soon)

## 🛠 Installation
```
git clone https://github.com/Tarek0/smbox.git
```
## 📄 License
smbox is released under the MIT License. See the LICENSE file for more details.

## 📮 Contact
For questions or feedback, please contact tareksalhi0@gmail.com

## 🚀 Usage Example
```
import pandas as pd
import openml

from smbox.Utils import Logger
from smbox.smbox_config import smbox_params
from smbox.ParamSpace import rf_default_param_space
from smbox.Optimise import Optimise

logger = Logger()

# Fetch a classification dataset from OpenML
dataset = openml.datasets.get_dataset(38)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
df = pd.DataFrame(X, columns=attribute_names)
df["target"] = y

# Basic data preprocesing 
y = df['target']
X = df.drop(target_name, axis=1)
X.fillna(0, inplace=True)

# Define a configuration dict to hold all key information
global config
config = {'dataset_source': 'openml'
    , 'dataset': '38'
    , 'algorithm': 'rf'
    , 'search_strategy': 'smbox'
    , 'search_strategy_config': smbox_params
    , 'wallclock': 600
    , 'output_root': './output/'
    }

data = {"X_train": X_train, "y_train":y_train} # requried data format for SMBOX

# use default rf hperparameter search space
cfg_schema = rf_default_param_space

logger.log(f'-------------Starting SMBOX')
optimiser = Optimise(config, random_seed=42)
best_parameters = optimiser.SMBOXOptimise(data, cfg_schema)
```
