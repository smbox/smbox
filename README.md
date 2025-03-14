# smbox

`smbox`:**Sequential Model-Based Optimization eXpress** is a lightweight Python library for Hyperparameter Optomisimation.

## Table of Contents

- [Introduction](#Introduction)
- [Installation](#Installation)
- [License](#License)
- [Contact](#Contact)
- [Example](#Example)
- [Cite](#Cite)

## 📖 Introduction
`smbox` provides a cutting-edge implementation of Sequential Model-Based Optimization (SMBO), tailored for tuning ML algorithm parameters. At its core, SMBO is designed to optimally explore expensive and noisy black-box functions, making it particularly suitable for hyperparameter optimization. With smbox, users can achieve peak performance in their ML models, bypassing the need for exhaustive and time-consuming search methods.

### Features:
🚂 State-of-the-art HPO: Easily achieve industry-leading hyperparameter optimization results.

🎯 Default Parameter Space: Enjoy plug-and-play convenience for the most popular ML algorithms. No need for defining grid search ranges or complex parameter distributions.

🎯 Custom Parameter Space: Tailor your parameter search space to your unique needs, or modify our default spaces with simplicity.

🤖️ Default Objective Function: Ideal for those using a natively supported ML algorithm.

🤖️ Custom Objective Function: Optimise any ML algorithm by selecting the performance metric that best fits your application.

## 🛠 Installation

### Using pip (for users)
If you just want to use smbox, you can install it directly from PyPI:
```bash
pip install smbox
```

Or install the latest version from GitHub:
```bash
pip install git+https://github.com/smbox/smbox.git
```

### Using Poetry (recommended for developers)
If you're a developer and intend to contribute or make changes to smbox, we recommend using Poetry for development:

1. First, install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/smbox/smbox.git
cd smbox
```

3. Install dependencies and set up development environment:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

### Development Tools
The project uses several development tools that are automatically installed with Poetry:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Static type checking
- **pytest**: Testing framework

To run the development tools:
```bash
# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run flake8 .

# Run type checking
poetry run mypy .

# Run tests
poetry run pytest
```

## 📄 License
smbox is released under the MIT License. See the LICENSE file for more details.

## 📮 Contact
For questions or feedback, please join our [Slack channel.](https://join.slack.com/t/slack-4aw5037/shared_invite/zt-22maoikro-_v_cxHvh7L_nMo7oqPvIvg)

## 🚀 Example
```
import pandas as pd
import openml
from smbox.utils import Logger
from smbox.optimise import Optimise
from smbox.smbox_config import smbox_params
from smbox.paramspace import rf_default_param_space
from smbox.default_objectives import rf_objective

# Fetch a classification dataset from OpenML
dataset = openml.datasets.get_dataset(38)
target_name ='target'

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
df = pd.DataFrame(X, columns=attribute_names)
df[target_name] = y

# Basic data preprocesing 
y = df[target_name]
X = df.drop(target_name, axis=1)
X.fillna(0, inplace=True)

# Configuration settings for the experiment.
# This dictionary holds key details for the setup, including dataset details, algorithm choice, search strategy, etc.
# Some keys enhance the clarity of logs and outputs, ensuring reproducibility and transparency in experiments.
global config
config = {
    'dataset_source': 'openml',               # Dataset's source platform; 'openml' in this instance.
    'dataset': 38,                            # Unique identifier for the dataset on OpenML.
    'algorithm': 'rf',                        # Chosen algorithm: Random Forest (denoted as 'rf').
    'search_strategy': 'smbox',               # Optimization/search strategy, specified as 'smbox'.
    'search_strategy_config': smbox_params,   # Configuration specifics for 'smbox'. Assumes `smbox_params` is predefined.
    'wallclock': 600,                         # Maximum time allotted for the task (600 seconds or 10 minutes).
    'output_root': './output/'                # Directory for saving output/results.
}

# Create a dictionary with training data. This format is needed for the SMBOX optimizer.
data = {"X_train": X, "y_train":y}

# Use our default hyperparameter search space for a Random Forest algorithm.
cfg_schema = rf_default_param_space

logger = Logger()
logger.log(f'-------------Starting SMBOX')
optimiser = Optimise(config, rf_objective, random_seed=42)
best_parameters, best_perf = optimiser.SMBOXOptimise(data, cfg_schema)
```
## Cite
```bibtex
@InProceedings{10.1007/978-3-031-53966-4_12,
author="Salhi, Tarek and Woodward, John",
title="SMBOX: A Scalable and Efficient Method for Sequential Model-Based Parameter Optimization",
booktitle="Machine Learning, Optimization, and Data Science",
year="2024",
pages="149--162",
isbn="978-3-031-53966-4"
}
```
