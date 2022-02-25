# cvas - Cross-Validation Framework

This cross-validation framework has the following goals:
1. Increase randomness across all aspects of cross-validation process (splits, 
random seeds of all components of the pipeline)
2. At the same time make experiments repeatable by fixing a global seed
3. Provide variance of the metric in addition to the mean
Variance  often provides more information about
model performance than the mean, but often excluded from the 
cross-validation frameworks.
4. Provide a simple hyperparameter tuning utility based on `optuna`

## Installation

Install from github:

`pip install git+https://github.com/mlarionov/cvas.git`

## Class `ModelGenerator`

This is an abstract class. Its subclasses will do two things:
1. Generate a training pipeline given the hyperparameters
2. Suggest new set of hyperparameters using `optuna`

Example:

```python
class LooModelGenerator(ModelGenerator):

    def suggest_parameters(self, trial: Trial):
        return {
            'loo__sigma': trial.suggest_loguniform('loo__sigma', 1E-5, 1E-1),
            'rf__max_depth': trial.suggest_int('rf__max_depth', 5, 40),
            'rf__max_features': trial.suggest_int('rf__max_features', 1, 10),
            'rf__min_samples_leaf': trial.suggest_int('rf__min_samples_leaf', 1, 3)}

    def get_model(self, parameters, random_state):
        """
        Get Leave One Out model
        :param random_state: random state
        :return: model
        """
        loo = LeaveOneOutEncoder(
            cols=self.cat_columns,
            random_state=self._compute_random_state(random_state, 0),
            sigma=parameters["loo__sigma"],
        )
        loo_rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=parameters['rf__max_depth'],
            max_features=parameters['rf__max_features'],
            min_samples_leaf=parameters['rf__min_samples_leaf'],
            random_state=self._compute_random_state(random_state, 1),
            n_jobs=-1)
        loo_pipe = Pipeline(steps=[('bbk', loo), ('rf', loo_rf)])
        return loo_pipe

```

## Class `KFoldCrossValidator`

The `validate()` method of this class trains models on a different KFold splits and returns
the mean and the standard deviation of the metric. This method uses `ModelGenerator` class

## Class `HyperParameterTuner`

This class creates an `optuna` study and remembers not only the best parameters, 
but also mean and standard deviation of the metric. In fact, it remembers
the results of all trials, allowing you not only select the best trial, 
but a set of good models that can be used in an ensemble.

Example:

```python
    cross_validator = KFoldCrossValidator(X_train, y_train, 3, 5, 8506, predictors.columns)
    tuner = HyperParameterTuner(cross_validator, model_generator)
    results_mean, results_stdev, params = tuner.study(35)
```

