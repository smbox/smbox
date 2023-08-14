import json
import pandas as pd
from typing import Dict, Any
import hashlib
from Utils import Logger

logger = Logger()  # Singleton

class ParamSpace:

    @staticmethod
    def create_cfg_schema(cfg_schema_path: str):
        """
        Loads the configuration schema from a given JSON file.

        This schema typically contains the set of hyperparameters to be optimized for an algorithm.

        Parameters:
        cfg_schema_path (str): The path to the JSON configuration file.

        Returns:
        dict: The configuration schema as a dictionary.
        """
        with open(cfg_schema_path) as json_file:
            cfg_schema = json.load(json_file)

        return cfg_schema

    @staticmethod
    def update_config_schema(cfg_schema: Dict, cfg_perf_hist: pd.DataFrame) -> Dict:
        """
        Updates the configuration schema based on the best configuration found so far.

        Parameters:
        cfg_schema (dict): The original configuration schema.
        population_fitness_history (pd.DataFrame): DataFrame containing the history of the population fitness.

        Returns:
        dict: The updated configuration schema.

        Raises:
        KeyError: If a parameter key is not found in the cfg_schema dictionary or in the population_fitness_history DataFrame.
        ValueError: If the population_fitness_history DataFrame does not contain a 'value' column.
        """
        if 'value' not in cfg_perf_hist.columns:
            raise ValueError("'value' column not found in population_fitness_history DataFrame.")

        if 'tune' not in cfg_schema:
            raise KeyError("'tune' key not found in cfg_schema dictionary.")

        params = list(cfg_schema['tune'].keys())
        cfg_perf_hist.sort_values('value', inplace=True, ascending=False)

        for param in params:
            if param not in cfg_perf_hist.columns:
                raise KeyError(f"'{param}' column not found in population_fitness_history DataFrame.")

            if cfg_schema['tune'][param]['type'] != 'str':
                # update mu
                cfg_schema['tune'][param]['mu'] = cfg_perf_hist.iloc[0][param]
                # update sigma
                cfg_schema['tune'][param]['sigma'] = cfg_schema['tune'][param]['mu'] * 3
            else:
                # update category choice to only include the best
                pivot = cfg_perf_hist.pivot_table(index=param, values='value', aggfunc='mean')
                best_cat_value = [pivot.sort_values(by='value', ascending=False).index.values[0]]
                cfg_schema['tune'][param]['categories'] = best_cat_value
                logger.log(f'Categotical feature {param} value fixed to - {best_cat_value}', 'DEBUG')

        return cfg_schema

    @staticmethod
    def feasiable_check(cfg_schema, population):
        """
        Verify the feasibility of the configurations in a population based on a given configuration schema.

        This function iterates over each configuration in the provided population. For each configuration,
        it checks every parameter against the constraints defined in the configuration schema. If a parameter
        value does not meet the constraints, it is adjusted to a feasible value. The function returns a list
        of feasible configurations.

        Parameters
        ----------
        cfg_schema : dict
            A dictionary defining the configuration schema. The schema should contain two keys: 'tune' and 'fix'.
            'tune' corresponds to a dictionary of parameters to be tuned, where each key-value pair is a parameter
            and its corresponding constraints (e.g., 'min', 'max', 'sample_dist', and 'categories' for categorical parameters).
            'fix' corresponds to a dictionary of parameters with fixed values.

        population : list of dict
            A list of configurations. Each configuration is a dictionary where keys are parameter names and values are
            corresponding parameter values.

        Returns
        -------
        list of dict
            A list of feasible configurations. Each configuration is a dictionary where keys are parameter names and values
            are corresponding feasible parameter values.
        """
        try:
            tune_params = list(cfg_schema['tune'].keys())
        except Exception as ex:
            tune_params = []
        try:
            fixed_params = list(cfg_schema['fix'].keys())
        except Exception as ex:
            fixed_params = []

        feasiable_population = []
        for i in range(0, len(population)):
            cfg_ = population[i]
            valid_cfg = {}

            for param in tune_params:
                if cfg_schema['tune'][param]['sample_dist'] != 'choice':
                    value = min(max(abs(cfg_[param]), cfg_schema['tune'][param]["min"]), cfg_schema['tune'][param]["max"])
                    valid_cfg[param] = value
                elif cfg_schema['tune'][param]['sample_dist'] == 'choice':
                    if cfg_[param] not in cfg_schema['tune'][param]['categories']:
                        valid_cfg[param] = cfg_schema['tune'][param]['categories'][0]
                    else:
                        valid_cfg[param] = cfg_[param]

            for param in fixed_params:
                valid_cfg[param] = cfg_[param]

            feasiable_population.append(valid_cfg)

        return feasiable_population

    @staticmethod
    def dict_hash(dictionary: Dict[str, Any]) -> str:
        """
        Compute the MD5 hash of a dictionary.

        This function computes the MD5 hash of a dictionary, taking into account the
        order of keys and values. The function first sorts the dictionary by key and
        then computes the hash. This ensures that equivalent dictionaries (i.e.,
        dictionaries with the same keys and values but different orderings) will
        produce the same hash.

        Parameters
        ----------
        dictionary : Dict[str, Any]
            The dictionary to compute the MD5 hash of.

        Returns
        -------
        str
            The MD5 hash of the input dictionary as a hexadecimal string.
        """
        dhash = hashlib.md5()
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

rf_default_param_space = {
   "tune":{
      "max_features":{
         "min":0.05,
         "max":0.95,
         "mu":0.5,
         "sigma":0.05,
         "sample_dist":"uniform",
         "type":"float"
      },
      "n_estimators":{
         "min":5,
         "max":500,
         "mu":100,
         "sigma":20,
         "sample_dist":"log_normal",
         "type":"int"
      },
      "max_depth":{
         "min":2,
         "max":100,
         "mu":20,
         "sigma":5,
         "sample_dist":"log_normal",
         "type":"int"
      },
      "min_samples_leaf":{
         "min":0.00001,
         "max":0.3,
         "mu":0.005,
         "sigma":0.005,
         "sample_dist":"uniform",
         "type":"float"
      },
      "min_samples_split":{
         "min":0.0001,
         "max":0.3,
         "mu":0.01,
         "sigma":0.005,
         "sample_dist":"uniform",
         "type":"float"
      },
      "class_weight":{
         "categories":[
            "balanced",
            "balanced_subsample"],
         "sample_dist":"choice",
         "type":"str"
      }
   }, "fix":{}
}

xgb_default_param_space = {
   "tune":{
      "n_estimators":{
         "min":2,
         "max":500,
         "mu":100,
         "sigma":50,
         "sample_dist":"log_normal",
         "type":"int"
      },
      "max_depth":{
         "min":50,
         "max":500,
         "mu":100,
         "sigma":25,
         "sample_dist":"log_normal",
         "type":"int"
      },
      "learning_rate":{
         "min":0.0001,
         "max":0.5,
         "mu":0.2,
         "sigma":0.02,
         "sample_dist":"uniform",
         "type":"float"
      },
      "booster":{
         "categories":[
            "gbtree",
            "gblinear",
            "dart"
         ],
         "sample_dist":"choice",
         "type":"str"
      },
      "gamma":{
         "min":0.0001,
         "max":0.2,
         "mu":0.1,
         "sigma":0.002,
         "sample_dist":"uniform",
         "type":"float"
      },
      "min_child_weight":{
         "min":0.0001,
         "max":0.2,
         "mu":0.1,
         "sigma":0.002,
         "sample_dist":"uniform",
         "type":"float"
      },
      "max_delta_step":{
         "min":0.0001,
         "max":0.5,
         "mu":0.2,
         "sigma":0.05,
         "sample_dist":"uniform",
         "type":"float"
      },
      "subsample":{
         "min":0.0001,
         "max":0.2,
         "mu":0.1,
         "sigma":0.002,
         "sample_dist":"uniform",
         "type":"float"
      },
      "colsample_bytree":{
         "min":0.0001,
         "max":0.5,
         "mu":0.1,
         "sigma":0.02,
         "sample_dist":"uniform",
         "type":"float"
      },
      "colsample_bylevel":{
         "min":0.0001,
         "max":0.5,
         "mu":0.1,
         "sigma":0.02,
         "sample_dist":"uniform",
         "type":"float"
      },
      "colsample_bynode":{
         "min":0.0001,
         "max":0.2,
         "mu":0.1,
         "sigma":0.02,
         "sample_dist":"uniform",
         "type":"float"
      },
      "reg_alpha":{
         "min":0.0001,
         "max":0.2,
         "mu":0.1,
         "sigma":0.002,
         "sample_dist":"uniform",
         "type":"float"
      }
   },
   "fix":{
      "scale_pos_weight":1.0
   }
}
