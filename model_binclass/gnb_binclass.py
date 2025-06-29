from sklearn.naive_bayes import GaussianNB
import utils.model_utils as mu
from model_binclass.binclass import BinClass

class GausianNBBinClass(BinClass):
    
    # === 1. LEARN ===
    def learn(self, df, features, target, params=None, search=True, cv=5, scoring='recall', random_state=42):
        """
    Train a Gaussian Naive Bayes classifier with optional hyper-parameter tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset that contains feature columns, the target column and a
        'dataset' indicator column (which is excluded from training).
    target_col : str
        Name of the target column to predict.
    params : dict, optional
        Hyper-parameters to evaluate or set. If `None`, the classifier is
        fitted with its default parameters.
    search : bool, default=True
        If True and `params` is provided, performs grid search over `params`;
        otherwise, fits directly with the supplied `params`.
    cv : int, default=5
        Number of cross-validation folds used during grid search.
    scoring : str, default='recall'
        Scoring metric for model selection when grid search is performed.
    random_state : int, default=42
        Kept for a unified API; not used by `GaussianNB`.

    Returns
    -------
    dict
        {
            'model'        : fitted `GaussianNB` instance,
            'model_params' : best or fixed parameter set as a dict,
            'features'     : None  # GaussianNB provides no intrinsic importances
        }

    Notes
    -----
    - Column `'dataset'` is always dropped before training.
    - If `params` is `None`, the model is trained once with default settings.
    - Grid search is delegated to `mu.grid_search`.
    """
        self.model = GaussianNB()
        self.features = features
        self.target = target

        X = df[features]
        y = df[target]

        if params:
            if search:
                # Use param grid if given
                self.best_params, self.best_model = mu.grid_search(self.model, X, y, params, cv=cv, scoring=scoring, n_jobs=-1)
            else:
                self.model.set_params(**params)
                self.best_model = self.model.fit(X, y)
                self.best_params = self.model.get_params()
        else:
            self.best_model = self.model.fit(X, y)
            self.best_params = self.model.get_params()

        return self                                                                                                                                                 