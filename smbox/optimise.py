import os
import json
import pandas as pd
import numpy as np
import math
import random
import time
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from smbox.utils import Logger
from smbox.paramspace import ParamSpace

logger = None

search_strategy_config_ = {'lf_init_ratio': 0.3
        , 'lf_init_n': 35
        , 'lf_ratio': 1.00
        , 'alpha_n': 2
        , 'inc_rand': 'Y'
        , 'inc_pseudo_rand': 'N'}

class Optimise:

    def __init__(self, config, random_seed, mlflow_tracking=False):
        global logger
        if logger is None:
            logger = Logger()

        self.config = config
        self.output_root = config['output_root']
        self.random_seed = random_seed
        self.mlflow_tracking = mlflow_tracking

        if mlflow_tracking:
            import mlflow

            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            # Create a new experiment
            experiment_name = f"smbox_{timestamp}"
            mlflow.create_experiment(experiment_name)

            # Get the experiment ID for the new experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

    def create_population(self, cfg_schema, population_size):
        """
        Create population of candiate hyper parameter configurations.
        :param cfg_schema. hyper parameterr schema
        :param population_size. the size of population to generate
        :return population. dictionary of population candiate configurations.
        """

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        try:
            tune_params_schema = cfg_schema['tune']
            tune_params = list(cfg_schema['tune'].keys())
        except Exception as ex1:
            tune_params = []

        try:
            fixed_params_schema = cfg_schema['fix']
            fixed_params = list(cfg_schema['fix'].keys())
        except Exception as ex:
            fixed_params = []

        population = []
        for i in range(0, population_size):
            cfg_ = {}
            try:
                for param in tune_params:
                    if tune_params_schema[param]['sample_dist'] == 'uniform':
                        value = random.uniform(tune_params_schema[param]["min"], tune_params_schema[param]["max"])
                    elif tune_params_schema[param]['sample_dist'] == 'log_normal':
                        value = math.log(
                            random.lognormvariate(tune_params_schema[param]["mu"], tune_params_schema[param]["sigma"]))
                    elif tune_params_schema[param]['sample_dist'] == 'choice':
                        # value = random.choice(tune_params_schema[param]["categories"], weights=tune_params_schema[param]["weights"] )
                        value = random.choice(tune_params_schema[param]["categories"])
                    if tune_params_schema[param]['type'] == 'int':
                        value = int(math.floor(value))
                    cfg_[param] = value

                for param in fixed_params:
                    cfg_[param] = fixed_params_schema[param]
            except Exception as ex:
                continue

            population.append(cfg_)

        return population


    def objective(self, cfg, data):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_validate
        """
        Evaluate the fitness of each solution of the population.
    
        Args:
            cfg (dict): Hyperparameter configuration space including bounds and types.
            data (dict): Dataset split into train and test sets.
    
        Returns:
            float: Fitness score.
    
        """
        if self.config['algorithm'] == 'xgb':
            model = XGBClassifier(**cfg, random_state=42)
        elif self.config['algorithm'] == 'rf':
            model = RandomForestClassifier(**cfg, random_state=42)
        else:
            raise ValueError("No algorithm provided.")

        cv_results = cross_validate(model,
                                    data['X_train'],
                                    data['y_train'],
                                    scoring='roc_auc',
                                    cv=4,
                                    return_train_score=True,
                                    return_estimator=True)

        perf = cv_results['test_score'].mean()

        return perf

    def evaluate_population(self, population, data, trial_counter=0):
        """
        Evaluate a set of hyperparameter configurations (also known as population).

        Args:
            population (list): A list of hyperparameter configurations.
            data (dict): Dataset split into train and test sets.
            trial_counter (int): A starting count of the number of configurations evaluated.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation results.
            str: Status indicating the time allowance (OK if within limit, END if exceeded).

        """

        _keys = list(population[0]) + ['value', 'number', 'time_elapsed']
        df = pd.DataFrame(columns=_keys)
        time_status = 'OK'

        for i in range(len(population)):
            if time.time() > t_end:  # Time allowance exceeded
                logger.log('Times up', 'DEBUG')
                time_status = 'END'
                break

            cfg = population[i]  # Parameters to be evaluated
            optimiser = Optimise(self.config, self.random_seed)
            perf = optimiser.objective(cfg, data)

            if self.mlflow_tracking:
                import mlflow
                with mlflow.start_run(experiment_id=self.experiment_id):
                    logger.log((cfg, perf), 'DEBUG')
                    mlflow.log_params(cfg)
                    mlflow.log_metric("perf", perf)
                    mlflow.log_metric("time", time.time())
                    mlflow.end_run()
                logger.log(f'mlflow experiment created: {self.experiment_id}', 'DEBUG')

            _row = list(population[i].values()) + [perf, i + trial_counter, time.time() - t_start]
            df.loc[i] = _row


        return df, time_status


    def create_lowfidelity_dataset(self, data, sample_ratio=0.05):
        """
        Create a low fidelity training dataset by downsampling the original dataset.

        Args:
            data (dict): Dataset split into train and test sets.
            sample_ratio (float): The downsampling ratio. Default is 0.05.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            dict: Dictionary containing the modified dataset with downscaled training data.

        """

        X_train, X_ignore, y_train, y_ignore = train_test_split(data["X_train"],
                                                                data["y_train"],
                                                                train_size=sample_ratio,
                                                                random_state=self.random_seed)

        data_temp = data.copy()
        data_temp["X_train"], data_temp["y_train"] = X_train, y_train  # Overwrite training data with downsampled

        return data_temp


    def format_best_trial_output(self, perf, best_params):
        """
        Format the best trial output in a standardized structure.

        Args:
            perf: Performance metric value of the best parameters.
            best_params: Dictionary containing the best hyperparameters found.

        Returns:
            pd.DataFrame: Pandas DataFrame containing the formatted output.

        """

        data_row = [
            self.config['dataset'],
            self.config['algorithm'],
            self.config['search_strategy'],
            best_params,
            perf,
            str(self.config),
            self.config['run_key']
        ]

        cols = [
            'dataset',
            'algorithm',
            'search_strategy',
            'param',
            'value',
            'full_config',
            'run_key'
        ]

        return pd.DataFrame(data=[data_row], columns=[cols])

    def format_trials_output(self, cfg_schema, input_df):
        df = input_df.copy()
        try:
            tune_params = list(cfg_schema['tune'].keys())
        except:
            tune_params = []
        try:
            fixed_params = list(cfg_schema['fix'].keys())
        except:
            fixed_params = []

        params = tune_params + fixed_params
        df['param'] = df[params].to_dict(orient='records')
        df['param_hash'] = df['param_hash'] = df['param'].apply(lambda row: ParamSpace.dict_hash(row))

        df['full_config'] = str(self.config)
        df['config_hash'] = ParamSpace.dict_hash(self.config)
        df['dataset'] = self.config['dataset']
        df['algorithm'] = self.config['algorithm']
        df['search_strategy'] = self.config['search_strategy']
        df['run_key'] = self.config['run_key']

        return df[
            ["dataset", "algorithm", "search_strategy", "param", "gen", "number", "time_elapsed", "value", "full_config",
             "param_hash", "config_hash", "run_key"]]


    def save_output(self, _df_trials=pd.DataFrame(), _df_holdout=pd.DataFrame(), wallclock_dict=None):
        """
        Save the output data to files.

        Args:
            output_root (str): The root directory where the output files will be saved.
            _df_holdout (pd.DataFrame): The DataFrame containing the best found params on the main output data. Defaults to an empty DataFrame
            _df_trials (pd.DataFrame, optional): The DataFrame containing trial data. Defaults to an empty DataFrame.
            wallclock_dict (dict, optional): Dictionary containing wall clock data. Defaults to None.

        Returns:
            None

        """

        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

        # Save trial data if provided
        if not _df_trials.empty:
            output_path = f"{self.output_root}trials_{self.config['search_strategy']}_{self.config['dataset']}_{self.config['algorithm']}_{timestamp}.csv"
            _df_trials.to_csv(output_path, index=False)
            logger.log(f'Trial output saved to: {output_path}')

        # Save holdout test set result data if provided
        if not _df_holdout.empty:
            output_path = f"{self.output_root}_run_log_holdout_{self.config['search_strategy']}_{self.config['dataset']}_{self.config['algorithm']}_{timestamp}.csv"
            _df_holdout.to_csv(output_path, index=False)
            logger.log(f'Output saved to: {output_path}')

        # Save wall clock data if provided
        if wallclock_dict is not None:
            output_path = f"{self.output_root}wallclock_{self.config['search_strategy']}_{self.config['dataset']}_{self.config['algorithm']}_{timestamp}.json"
            with open(output_path, 'w') as fp:
                json.dump(wallclock_dict, fp)
                logger.log(f'Wall clock data saved to: {output_path}')


    def fit_response_surface_model(self, cfg_schema, population_fitness_history, params):
        """
            Fits a response surface model using CatBoostRegressor to predict the fitness value of a given set of hyperparameters.

            Args:
                cfg_schema (Dict): A dictionary containing the schema of the configuration space.
                population_fitness_history (pd.DataFrame): A pandas DataFrame containing the history of evaluated configurations and their corresponding fitness values.
                params (List[str]): A list of strings representing the names of the hyperparameters to be used for training the model.

            Returns:
                cb (CatBoostRegressor): The trained CatBoostRegressor model.
        """
        X = population_fitness_history[params]
        y = population_fitness_history['value']
        logger.log(X.describe(), 'DEBUG')

        cat_features = []
        for k in list(cfg_schema['tune'].keys()):
            if cfg_schema['tune'][k]['type'] == 'str':
                cat_features.append(k)

        cb = CatBoostRegressor(thread_count=-1, verbose=False)
        cb.fit(X, y
               , cat_features=cat_features
               , plot=False)

        return cb

    @staticmethod
    def _mutation(cfg_schema, _gene, mutation_rate, alpha=0.10):
        """mutate each gene based on the given mutation rate."""

        gene = _gene.copy()
        for i in range(len(gene)):
            x = np.random.random(1)[0]
            if x < mutation_rate:
                if isinstance(gene[i], int) == True:
                    alpha_shift = max(np.floor(gene[i] * alpha), 1)
                    # print(-alpha_shift, alpha_shift)
                    random_value = np.random.randint(-alpha_shift, alpha_shift)
                    gene[i] = (gene[i] + random_value)
                    gene[i] = max(gene[i], 1)  # no parameters can have zero or negative values
                elif isinstance(gene[i], float) == True:
                    alpha_shift = gene[i] * alpha
                    random_value = random.uniform(-alpha_shift, alpha_shift)
                    # random_value = random.uniform(-0.0001,0.0001)
                    gene[i] = (gene[i] + random_value)
                elif isinstance(gene[i], str) == True:
                    list_ = cfg_schema['tune'][gene.index[i]][
                        'categories']  # look up possible values for the categorical param
                    gene[i] = random.choice(list_)

        return gene

    @staticmethod
    def anchor_new_cfgs(cfg_schema, anchor_cfgs, generate_n=2, expore_alpha=0.25):
        """
        Generate new configurations near the anchor configuration.

        Args:
            cfg_schema (dict): The schema defining the configuration structure.
            anchor_cfgs (pd.DataFrame): DataFrame containing anchor configurations.
            generate_n (int, optional): The number of new configurations to generate for each anchor. Defaults to 2.
            expore_alpha (float, optional): The alpha value for exploration during mutation. Defaults to 0.25.

        Returns:
            pd.DataFrame: DataFrame containing the generated new configurations.

        """

        new_cfgs = pd.DataFrame()

        for anchor_i in range(len(anchor_cfgs)):
            anchor = anchor_cfgs.iloc[anchor_i]
            new_cfgs_list = []

            for i in range(generate_n):
                _cfg = Optimise._mutation(cfg_schema, anchor, mutation_rate=1.0, alpha=expore_alpha).copy()
                new_cfgs_list.append(_cfg)

            new_cfgs = new_cfgs.append(pd.DataFrame(new_cfgs_list))

        # Add fixed param values
        for key, value in cfg_schema['fix'].items():
            new_cfgs[key] = value

        return new_cfgs


    def SMBOXOptimise(self, data_all, cfg_schema, save_trials=True):
        """
        This function performs the SMBOX search routine experiment. SMBOX is a strategy to
        optimize hyperparameters of machine learning models using sequential model-based
        optimization (SMBO).

        This function takes a configuration dictionary and a random seed as inputs,
        and performs a series of operations including data preparation, schema creation,
        initialization, and iterative model training and optimization. It also handles the
        creation and evaluation of low fidelity datasets and manages the search strategy
        for hyperparameters.

        Parameters:
        config (dict): A dictionary containing key configuration options for the experiment.
                       It should include keys such as 'search_strategy', 'wallclock', 'dataset',
                       'algorithm', and others that control various aspects of the experiment.
        data (dict): A dictionary containing the training and test datset.

        _random_seed (int): A seed for the random number generator used in the experiment.
                            This is used to ensure the reproducibility of the experiment.

        The function ends by saving output and indicating completion of the run.
        """
        try:
            wallclock_seconds = self.config['wallclock']
            logger.log(f"Starting run for: {self.config['dataset']}, for {wallclock_seconds} seconds")
            search_strategy_config_ = self.config['search_strategy_config']

            params = list(cfg_schema['tune'].keys())
            fixed_params = list(cfg_schema['fix'].keys())
            all_params = params + fixed_params  # get a list of all cfg params
            logger.log(f' Tuning parameters: {params}', 'DEBUG')

            global t_end
            global t_start
            t_start = time.time()
            t_end = t_start + wallclock_seconds  # set end time
            self.config['run_key'] = f"{self.config['search_strategy']}_{self.config['dataset']}_{self.config['algorithm']}_{t_start}"

            # Initialization
            logger.log('Initialization - Random search to train a response surface model', 'DEBUG')
            if search_strategy_config_['lf_init_ratio'] == 1.0:
                data_low_fidelity = data_all.copy()
            else:
                data_low_fidelity = self.create_lowfidelity_dataset(data_all, search_strategy_config_['lf_init_ratio'],)
            population_candidates = self.create_population(cfg_schema, search_strategy_config_['lf_init_n'])
            population = ParamSpace.feasiable_check(cfg_schema, population_candidates)
            if not population:
                raise IndexError(f"The population list is empty! population_candidates: {population_candidates}")
            population_fitness, time_status = self.evaluate_population(population, data_low_fidelity)
            logger.log('Completed initialization', 'DEBUG')
            if search_strategy_config_['lf_ratio'] == 1.00:
                data = data_all
            else:
                logger.log('Sampling to generate low fidelity training dataset', 'DEBUG')
                data = self.create_lowfidelity_dataset(data_all, search_strategy_config_['lf_ratio'])

            # Create history table
            population_fitness_history = population_fitness.copy()
            population_fitness_history["gen"] = 0
            global_best = population_fitness.value.max()
            logger.log(f'Global best so far: {global_best}')

            # Update hp configuration schema
            cfg_schema = ParamSpace.update_config_schema(cfg_schema, population_fitness_history)
            logger.log('Updated cfg_schema', 'DEBUG')
            logger.log(cfg_schema, 'DEBUG')

            gen = 0
            while time_status == 'OK':
                gen += 1
                # Fitting response model
                regressor = self.fit_response_surface_model(cfg_schema, population_fitness_history, params)
                # Identify best population candidates
                population_candidates = self.create_population(cfg_schema, 50000)
                population_candidates = ParamSpace.feasiable_check(cfg_schema, population_candidates)
                df_population_candidates = pd.DataFrame(population_candidates)
                # Check against cache to remove any already calculated params configurations
                temp_cache = pd.concat([df_population_candidates, population_fitness_history[all_params]], axis=0)

                df_population_candidates = temp_cache[~temp_cache.duplicated(subset=all_params, keep=False)].copy()
                # Predict fitness of each param cfg
                df_population_candidates['value'] = regressor.predict(df_population_candidates[params])
                df_population_candidates.sort_values('value', inplace=True, ascending=False)

                # Inject random param cfgs (meta parameter)
                if search_strategy_config_['inc_rand'] == 'Y':
                    logger.log('Including random cfgs', 'DEBUG')
                    rand = random.randint(search_strategy_config_['alpha_n'], len(df_population_candidates) - 1)
                    df_best_candidates = pd.concat([
                        df_population_candidates.head(search_strategy_config_['alpha_n']),
                        df_population_candidates.iloc[[rand]]
                    ])
                else:
                    df_best_candidates = df_population_candidates.head(search_strategy_config_['alpha_n'])
                df_best_candidates = df_best_candidates.loc[:, df_population_candidates.columns != 'value']
                df_best_candidates['param'] = df_best_candidates.to_dict(orient='records')
                population = list(df_best_candidates['param'])

                # Inject inc_pseudo random param cfgs (meta parameter)
                if search_strategy_config_['inc_pseudo_rand'] == 'Y':
                    logger.log('Generating pseudo random cfgs', 'DEBUG')
                    anchor_cfgs = population_fitness_history[params].head(2)
                    df_pseudo_rand_population_candidates = Optimise.anchor_new_cfgs(cfg_schema, anchor_cfgs, generate_n=1,
                                                                          expore_alpha=0.25)
                    df_pseudo_rand_population_candidates['param'] = df_pseudo_rand_population_candidates.to_dict(orient='records')
                    pseudo_rand_population = list(df_pseudo_rand_population_candidates['param'])
                    population = population + pseudo_rand_population

                # Evaluate the best
                population_fitness, time_status = self.evaluate_population(population, data,
                                                                      trial_counter=len(population_fitness_history))
                population_fitness["gen"] = gen

                # Update history and global best
                population_fitness_history = pd.concat([population_fitness_history, population_fitness], ignore_index=True)
                challenger = population_fitness.value.max()
                if challenger > global_best:
                    logger.log(f'improvement: {max(challenger - global_best, 0)}')
                    global_best = challenger
                    logger.log(f'Global best so far: {global_best}')

            logger.log(f'Global best: {global_best}')
            best_params_df = population_fitness_history[population_fitness_history.value == global_best].head(1)
            best_params = best_params_df.drop(['value', 'number', 'gen', 'time_elapsed'], axis=1).iloc[0].to_dict()
            logger.log(f'Best params: {best_params}')
            #test_perf = calculate_test_set_performance(data, best_params)
            #log(f'Test set performance: {test_perf}')

            if save_trials:
                df_trials = self.format_trials_output(cfg_schema, population_fitness_history)
                #df_holdout = optimiser.format_best_trial_output(test_perf, best_params)
                #optimiser.save_output(self.output_root, _df_trials=df_trials, _df_holdout=df_holdout)
                self.save_output(_df_trials=df_trials)

            logger.log('RUN COMPLETE')

            return best_params, global_best

        except IndexError as e:
            logger.log(f"Error encountered: {e}")