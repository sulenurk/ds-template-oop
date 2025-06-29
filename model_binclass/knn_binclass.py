from sklearn.neighbors import KNeighborsClassifier
import utils.model_utils as mu
from model_binclass.binclass import BinClass

class KnnBinClass(BinClass):
    
    # === 1. LEARN ===
    def learn(self, df, features, target, params=None, search=True, cv=5, scoring='recall', random_state=42):
        """
        Train a K-Nearest Neighbors classifier with optional hyper-parameter tuning.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing feature columns, the target column, and a
            'dataset' indicator column that is excluded from training.
        target_col : str
            Name of the target column to be predicted.
        params : dict, optional
            Hyper-parameters to evaluate or set. If None, a predefined parameter
            grid is used for tuning.
        search : bool, default=True
            If True and `params` is provided, performs grid search using `params`.
            If False, fits the model directly with the provided `params`.
        cv : int, default=5
            Number of cross-validation folds used during grid search.
        scoring : str, default='recall'
            Scoring metric for model selection in grid search.
        random_state : int, default=42
            Included for API consistency; KNN is deterministic and ignores it.

        Returns
        -------
        dict
            {
                'model'        : fitted `KNeighborsClassifier`,
                'model_params' : best or fixed parameter set as a dict
            }

        Notes
        -----
        - Column 'dataset' is always dropped before training.
        - Grid search is delegated to `mu.grid_search`.
        """

        self.model = KNeighborsClassifier()
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
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # p=1: Manhattan, p=2: Euclidean
            }
            self.best_params, self.best_model = mu.grid_search(self.model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

        return self
