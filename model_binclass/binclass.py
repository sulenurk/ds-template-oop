import pandas as pd
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib

class BinClass():

    def __init__(self):
        self.features = None
        self.target = None
        self.model = None
        self.best_model = None
        self.best_params = None
        self.shap_values = None
        self.metrics = None 
        self.feature_imp = None

    def learn(self):
        pass

    # === 2. APPLY ===
    def apply(self, df):
        """
        Apply a trained XGBoost model to a DataFrame and append predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Data on which predictions are required.
        model_info : dict
            Dictionary with key ``'model'`` mapping to a fitted ``XGBClassifier``.
        columns : list[str], optional
            Column names to exclude during prediction (e.g., IDs). They are
            re-attached to the returned DataFrame unchanged.

        Returns
        -------
        pd.DataFrame
            Copy of the input data (excluding any temporarily dropped columns
            during inference) with two new columns:  
            - ``'predictions'``   : predicted class labels  
            - ``'probabilities'`` : probability of the positive class (index 1)

        Notes
        -----
        - Existing ``'predictions'`` or ``'probabilities'`` columns in *df* are
        removed before new values are added.
        - Positive-class probability is extracted as
        ``model.predict_proba(... )[:, 1]``.
        """

        # Predict
        df['predictions'] = self.best_model.predict(df[self.features])
        df['probabilities'] = self.best_model.predict_proba(df[self.features])[:, 1]

        return df

    # === 3. EVALUATE ===
    def evaluate(self, df):
        """
        Evaluate classification results separately for training and test datasets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing:
            - true labels in *target*
            - a 'predictions' column (required)
            - a 'probabilities' column (optional) for probability-based plots
            - a 'dataset' column with values 1 for training and 0 for test
        target : str
            Name of the column holding true class labels.

        Returns
        -------
        None
        """
        train_set = df[df['dataset'] == 1]
        test_set = df[df['dataset'] == 0]

        #Train Set Evaluation
        if train_set.empty:
            print("No data found for the training set.")
            return

        y_true = train_set[self.target]
        y_pred = train_set['predictions']
        pred_proba = train_set['probabilities'] if 'probabilities' in train_set.columns else None

        print("\n═══ Train Set Classification Report ═══")
        print(classification_report(y_true, y_pred))
        mu.plot_confusion_matrix(y_true, y_pred)

        if pred_proba is not None:
            mu.plot_probability_metrics(y_true, pred_proba)
            mu.plot_lift_curve(y_true, pred_proba)
            mu.plot_cumulative_gains(y_true, pred_proba)
            mu.plot_calibration_curve(y_true, pred_proba)
        else:
            print("Note: Train set probability-based plots skipped (no probabilities found).")

        #Test Set Evaluation
        if test_set.empty:
            print("No data found for the test set.")
            return

        y_true = test_set[self.target]
        y_pred = test_set['predictions']
        pred_proba = test_set['probabilities'] if 'probabilities' in test_set.columns else None

        print("\n═══ Test Set Classification Report ═══")
        print(classification_report(y_true, y_pred))
        mu.plot_confusion_matrix(y_true, y_pred)

        if pred_proba is not None:
            mu.plot_probability_metrics(y_true, pred_proba)
            mu.plot_lift_curve(y_true, pred_proba)
            mu.plot_cumulative_gains(y_true, pred_proba)
            mu.plot_calibration_curve(y_true, pred_proba)
        else:
            print("Note: Test set probability-based plots skipped (no probabilities found).")    

    # === 4. EXPLAIN ===
    def explain(self, df, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
        """
        Generate SHAP-based model explanations including global and instance-level visualizations.

        Filters the input DataFrame to `dataset == 0` (inference set),
        computes SHAP values, and creates:
        - Global feature importance plots
        - Per-sample SHAP explanation (if `sample_index` is set)
        - SHAP feature plots for each instance (if `index_feature=True`)

        Parameters
        ----------
        model : fitted model object
            Trained model compatible with SHAP explainers.
        df : pd.DataFrame
            Input dataset containing a 'dataset' column. Only rows where
            `dataset == 0` are used.
        top_n_features : int, default=10
            Number of top features to display in plots.
        sample_index : int, optional
            Index of a specific sample for which SHAP explanations will be drawn.
        index_feature : bool, default=False
            If True, creates SHAP feature-wise plots for each observation.
        save_path : str, optional
            Directory path to save generated plots as images.

        Returns
        -------
        np.ndarray or list of np.ndarray
            SHAP values computed for the filtered DataFrame.

        Notes
        -----
        - Requires `sp` module to define `shap_values`, `global_analysis`,
        `index_charts`, and `index_feature` functions.
        - Assumes binary or multiclass classification.
        """
        df = df[df.dataset == 0][self.features]
        #Comments take place here
        self.shap_values = sp.shap_values(self.best_model, df)

        sp.global_analysis(self.shap_values, df, top_n_features=top_n_features, save_path=save_path)

        if sample_index is not None:
            sp.index_charts(self.shap_values, sample_index=sample_index, top_n_features=top_n_features, save_path=save_path)

        if index_feature:
            sp.index_feature(self.shap_values, df, save_path=save_path)

    # === 5. SAVE ===
    def save(self, filepath = None):
        """
        Save a trained model and its metadata to disk using joblib.

        Parameters
        ----------
        model_info : dict
            Dictionary containing the trained model and related metadata such
            as best hyperparameters, feature names, etc.
        filepath : str
            Full file path (including `.pkl` or `.joblib`) where the model will be saved.

        Returns
        -------
        None
            Prints confirmation to stdout upon successful save.
        """
        model_info = dict(model = self.model)
        joblib.dump(model_info, filepath)
        print(f"Model saved to {filepath}")

    # === 6. LOAD ===
    def load(self, filepath):
        """
        Load a previously saved model and its metadata from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file (produced by `save_model`).

        Returns
        -------
        dict
            A dictionary containing the loaded model object and its associated metadata.

        Notes
        -----
        - Prints a confirmation message when the model is successfully loaded.
        """
        model_info = joblib.load(filepath)
        self.model = model_info["model"]
        print(f"Model loaded from {filepath}")
        return model_info