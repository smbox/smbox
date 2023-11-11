from sklearn.model_selection import cross_validate
import time
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import check_scoring

def cross_val_performance_legacy(classifier, cfg, data, scoring='roc_auc', cv=4):
    """Evaluate model performance using cross-validation.

    Args:
        classifier (object): A classifier object that supports the fit method.
        cfg (dict): Hyperparameter configuration for the classifier.
        data (dict): Dataset split into train and test sets.
        scoring (str, optional): Scoring method for cross-validation. Defaults to 'roc_auc'.
        cv (int, optional): Number of cross-validation folds. Defaults to 4.

    Returns:
        float: Mean test score from cross-validation.
    """
    model = classifier(**cfg, random_state=42)

    cv_results = cross_validate(model,
                                data['X_train'],
                                data['y_train'],
                                scoring=scoring,
                                cv=cv,
                                return_train_score=True,
                                return_estimator=True)

    return cv_results['test_score'].mean()

def cross_val_performance(classifier, cfg, data, scoring='roc_auc', cv=4, time_limit=None):
    """Evaluate model performance using cross-validation with a time check between each fold.

    Args:
        classifier (object): A classifier object that supports the fit method.
        cfg (dict): Hyperparameter configuration for the classifier.
        data (dict): Dataset split into train and test sets.
        scoring (str, optional): Scoring method for cross-validation. Defaults to 'roc_auc'.
        cv (int, optional): Number of cross-validation folds. Defaults to 4.
        time_limit (int, optional): Time limit in seconds for the whole cross-validation process.

    Returns:
        float: Mean test score from cross-validation or None if the time limit is exceeded.
    """
    X_train = data['X_train']
    y_train = data['y_train']
    kf = KFold(n_splits=cv)
    scores = []
    scorer = check_scoring(classifier(), scoring=scoring)
    time_status = 'OK'  # Initialize time status
    start_time = time.time()
    #print(f'time_limit: {time_limit}', f'end_time: {start_time + time_limit}', f'current_time: {time.time()}')

    for train_index, test_index in kf.split(X_train):
        if time.time() > start_time + time_limit:
            time_status = 'END'
            break  # Time limit exceeded, break out of the loop

        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model = clone(classifier(**cfg, random_state=42))
        model.fit(X_train_fold, y_train_fold)
        score = scorer(model, X_test_fold, y_test_fold)
        scores.append(score)

    if time_status == 'END':
        return 0, time_status  # Return 0 if time limit was exceeded
    else:
        return sum(scores) / len(scores), time_status  # Otherwise, return the average score

def rf_objective(cfg, data, time_limit=None):
    """Evaluate the fitness of each solution of the population using RandomForest.

    Args:
        cfg (dict): Hyperparameter configuration space including bounds and types.
        data (dict): Dataset split into train and test sets.

    Returns:
        float: Fitness score.
    """
    from sklearn.ensemble import RandomForestClassifier

    return cross_val_performance(RandomForestClassifier, cfg, data, time_limit=time_limit)


def xgb_objective(cfg, data, time_limit=None):
    """Evaluate the fitness of each solution of the population using XGBoost.

    Args:
        cfg (dict): Hyperparameter configuration space including bounds and types.
        data (dict): Dataset split into train and test sets.

    Returns:
        float: Fitness score.
    """
    from xgboost import XGBClassifier

    return cross_val_performance(XGBClassifier, cfg, data, time_limit=time_limit)
