from sklearn.ensemble import GradientBoostingClassifier
import utils.model_utils as mu
from model_binclass.binclass import BinClass

class GradientBoostingBinclass(BinClass):

    # === 1. LEARN ===
    def learn(self, df, features, target, params=None, search=True, cv=5, scoring='recall', random_state=42):
        """
        Train a Gradient Boosting model with optional hyperparameter tuning.

        This function trains a `GradientBoostingClassifier` on the given DataFrame.
        It supports custom parameter input or automatic grid search for tuning.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing features, the target column, and a 'dataset' column that will be excluded.
        target_col : str
            Name of the target column to be predicted.
        params : dict, optional
            Dictionary of hyperparameters to use for training. If None, default parameter grid will be used.
        search : bool, default=True
            If True and `params` is provided, performs grid search using the provided `params`.
            If False, fits the model directly using `params`.
        cv : int, default=5
            Number of cross-validation folds used during hyperparameter search.
        scoring : str, default='recall'
            Scoring metric used for selecting the best model during grid search.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict
            A dictionary with:
            - 'model' (GradientBoostingClassifier): The trained model.
            - 'model_params' (dict): The best parameter set used for training.

        Notes
        -----
        - The column 'dataset' is excluded before training.
        - If no `params` are provided, a default grid is used for hyperparameter search.
        - Assumes binary or multiclass classification.
        """
        self.model = GradientBoostingClassifier(random_state=random_state)
        self.features = features
        self.target = target

        X = df[features]
        y = df[target]

        if params:
            if search:
                self.best_params, self.best_model = mu.grid_search(self.model, X, y, params, cv=cv, scoring=scoring, n_jobs=-1)
            else:
                self.model.set_params(**params)
                self.best_model = self.model.fit(X, y)
                self.best_params = params
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            self.best_params, self.best_model = mu.grid_search(self.model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

        return self