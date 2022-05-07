from abc import abstractmethod
from statistics import mean, stdev
from typing import Tuple, Literal, Callable, List

import numpy as np
import optuna
import pandas as pd
from numpy.typing import ArrayLike
from optuna import Trial
from sklearn.model_selection import StratifiedKFold, KFold


class ModelGenerator:

    def __init__(self, cat_columns):
        """
        Generate a model with a randomized random seed
        :param cat_columns: columns (we need it to create a DataFrame)
        """
        self.cat_columns = cat_columns

    @staticmethod
    def _compute_random_state(initial_state: int, instance: int):
        """
        Compute a new random seed
        :param initial_state: initial state
        :param instance: instance number
        :return: new random seed
        """
        return (initial_state % 153) * 41 + instance * 11

    @abstractmethod
    def get_model(self, parameters, random_state):
        pass

    @abstractmethod
    def suggest_parameters(self, trial: Trial):
        pass


class KFoldCrossValidator:

    def __init__(self,
                 X: ArrayLike,
                 y: ArrayLike,
                 n_splits: int,
                 n_repeats: int,
                 random_state: int,
                 metric: Callable[[ArrayLike, ArrayLike], float],
                 stratify: Literal["yes", "no", "auto"] = "auto",
                 columns=None):

        """
        Perform cross-validation
        :param X: predictor variables (most likely, a data frame)
        :param y: target variable
        :param n_splits: number of splits in K-Fold
        :param n_repeats: number of times we repeat K-Fold validation for a different random state
        :param random_state: initial random state
        :param stratify: Yes, no, auto: indicates if we need to stratify when doing a K-fold split
            by default we enable it for a binary classification problem
        :param columns: use these to create pandas DataFrames
        :return: an array with the accuracy results
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.columns = columns
        self.metric = metric
        is_binary_classification = len(np.unique(self.y)) < 3
        if stratify == "yes" or stratify == "auto" and is_binary_classification:
            self.generator_function = StratifiedKFold
        elif stratify == "no" or stratify == "auto" and not is_binary_classification:
            self.generator_function = KFold


    def validate(self, model_generator: ModelGenerator, parameters) -> List[float]:
        results = []
        for state in range(self.random_state, self.random_state + self.n_repeats):
            generator = self.generator_function(n_splits=self.n_splits, shuffle=True, random_state=state)
            for train_index, test_index in generator.split(self.X, self.y):
                X_arr = np.array(self.X)
                X_train, X_test = X_arr[train_index], X_arr[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                if self.columns is not None:
                    X_train = pd.DataFrame(X_train, columns=self.columns)
                    X_test = pd.DataFrame(X_test, columns=self.columns)

                model = model_generator.get_model(parameters, state)
                model.fit(X_train, y_train)
                results.append(self.metric(y_test, model.predict(X_test)))
        return results


class HyperParameterTuner:

    def __init__(
            self,
            cross_validator: KFoldCrossValidator,
            model_generator: ModelGenerator,
            direction: Literal["maximize", "minimize"] = "maximize"
    ):
        """
        Tune hyperparameters
        :param cross_validator: an instance of cross-validator
        :param model_generator: an instance of model generator
        :param direction: (maximize or minimize) direction to use when optimizing the objective:
        """
        self.cross_validator = cross_validator
        self.model_generator = model_generator
        self.direction = direction
        self.trial_results = {}

    def objective(self, trial: Trial):
        """
        Run the trial one time, average the metric
        :param trial: Optuna trial
        :return: mean of the metric
        """
        parameters = self.model_generator.suggest_parameters(trial)

        results = self.cross_validator.validate(self.model_generator, parameters)
        self.trial_results[trial.number] = results
        return mean(results)

    def study(self, n_trials: int) -> Tuple[float, float, dict]:
        """
        Conducts an Optuna study
        :param n_trials: Number of trials
        :return: mean and std of the metric for the best model.
        """
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=n_trials, )
        return study.best_trial.value, stdev(self.trial_results[study.best_trial.number]), study.best_params
