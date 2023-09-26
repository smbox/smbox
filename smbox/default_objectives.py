from sklearn.model_selection import cross_validate


def cross_val_performance(classifier, cfg, data, scoring='roc_auc', cv=4):
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


def rf_objective(cfg, data):
    """Evaluate the fitness of each solution of the population using RandomForest.

    Args:
        cfg (dict): Hyperparameter configuration space including bounds and types.
        data (dict): Dataset split into train and test sets.

    Returns:
        float: Fitness score.
    """
    from sklearn.ensemble import RandomForestClassifier

    return cross_val_performance(RandomForestClassifier, cfg, data)


def xgb_objective(cfg, data):
    """Evaluate the fitness of each solution of the population using XGBoost.

    Args:
        cfg (dict): Hyperparameter configuration space including bounds and types.
        data (dict): Dataset split into train and test sets.

    Returns:
        float: Fitness score.
    """
    from xgboost import XGBClassifier

    return cross_val_performance(XGBClassifier, cfg, data)
