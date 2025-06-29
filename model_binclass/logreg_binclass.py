import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils.model_utils as mu
from model_binclass.binclass import BinClass

class LogregBinClass(BinClass):
    
    # === 1. LEARN ===
    def learn(self, df, features, target, params=None, search=True, cv=5, scoring='recall', random_state=42):
        """
        Train a Logistic Regression classifier with optional hyper-parameter tuning.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing feature columns, the target column, and a
            'dataset' indicator column (excluded from training).
        target_col : str
            Name of the target column to predict.
        params : dict, optional
            Hyper-parameters to evaluate or set. If ``None``, a predefined grid is
            used for tuning.
        search : bool, default=True
            If ``True`` and ``params`` is provided, performs grid search over those
            parameters; if ``False``, fits directly with the supplied ``params``.
        cv : int, default=5
            Number of cross-validation folds used during grid search.
        scoring : str, default='recall'
            Score metric for model selection in grid search.
        random_state : int, default=42
            Random seed for reproducibility (passed to ``LogisticRegression``).

        Returns
        ------- 
        dict
            {
                'model'        : fitted ``LogisticRegression``,
                'model_params' : best or fixed parameter set as a ``dict``
            }

        Notes
        -----
        - Column ``'dataset'`` is always dropped before training.
        - Grid search is performed via ``mu.grid_search``.
        """
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
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
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            self.best_params, self.best_model = mu.grid_search(self.model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

        return self