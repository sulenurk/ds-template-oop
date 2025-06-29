from sklearn.tree import DecisionTreeClassifier
import utils.model_utils as mu
from model_binclass.binclass import BinClass

class DecisionTreeBinClass(BinClass):
    # === 1. LEARN ===
    def learn(self, df, features, target, params=None, search=True, cv=5, scoring='recall', random_state=42):
        """
        Train a Decision Tree model with optional hyperparameter tuning.

        This function trains a `DecisionTreeClassifier` on the provided DataFrame.
        It supports manual parameter input or automatic grid search for hyperparameter optimization.

        Parameters:
            df (pd.DataFrame): Input dataset containing features, target column, and a 'dataset' indicator column.
            target_col (str): Name of the target column to be predicted.
            params (dict, optional): Dictionary of hyperparameters to set for the model. If None, default grid search is used.
            search (bool, default=True): If True and `params` is provided, performs grid search over the given `params`. 
                                        If False, trains directly with provided `params`.
            cv (int, default=5): Number of cross-validation folds used in grid search.
            scoring (str, default='recall'): Scoring metric used for model selection during hyperparameter tuning.
            random_state (int, default=42): Random seed for reproducibility.

        Returns:
            dict: A dictionary containing:
                - 'model' (DecisionTreeClassifier): The trained decision tree model.
                - 'model_params' (dict): The best parameters used during training.

        Notes:
            - The column named 'dataset' is dropped from `df` before training.
            - If `params` is not provided, a predefined hyperparameter grid is used for tuning.
        
        """ 
        self.model = DecisionTreeClassifier(random_state=random_state)
        self.target = target
        self.features = features

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
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            self.best_params, self.best_model = mu.grid_search(self.model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

        return self