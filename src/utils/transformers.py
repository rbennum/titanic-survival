import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rare_threshold_reference="Master"):
        self.rare_threshold_reference = rare_threshold_reference
        self.rare_titles_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()
        titles = X_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        titles = titles.replace({"Ms": "Miss", "Mlle": "Miss", "Mme": "Mrs"})
        counts = titles.value_counts()
        ref_count = counts.get(self.rare_threshold_reference, 0)
        self.rare_titles_ = counts[counts < ref_count].index.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        titles = X_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        titles = titles.replace({"Ms": "Miss", "Mlle": "Miss", "Mme": "Mrs"})

        if self.rare_titles_:
            titles = titles.replace(self.rare_titles_, "Rare")

        known_titles = ["Mr", "Mrs", "Miss", "Master", "Rare"]
        titles = titles.where(titles.isin(known_titles), "Rare")
        X_df["Title"] = titles
        return X_df

    def get_feature_names_out(self, input_features=None):
        features = (
            list(input_features)
            if input_features is not None
            else (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else []
            )
        )
        if "Title" not in features:
            features.append("Title")
        return np.array(features, dtype=object)


class AgeImputer(BaseEstimator, TransformerMixin):
    """
    Example:

    from sklearn.pipeline import Pipeline

    1. Build a clean, flat pipeline
    ```
    age_title_pipeline = Pipeline([
        ('title_generator', TitanicTitleTransformer()),
        ('age_imputer', TitleGroupedAgeImputer())
    ])
    ```

    2. Tell scikit-learn to keep data as a Pandas DataFrame throughout the steps
    `age_title_pipeline.set_output(transform="pandas")`

    3. Execute
    Input: DataFrame with ['Age', 'Name']
    Output: DataFrame with ['Age', 'Name', 'Title'] (with Age fully imputed!)

    `processed_df = age_title_pipeline.fit_transform(df_train[['Age', 'Name']])`
    """

    def __init__(self):
        self.global_median_age_ = None
        self.medians_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.global_median_age_ = X_df["Age"].median()
        self.medians_ = X_df.groupby("Title")["Age"].median().to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        title_medians = X_df["Title"].map(self.medians_)
        title_medians = title_medians.fillna(self.global_median_age_)
        X_df["Age"] = X_df["Age"].fillna(title_medians)
        return X_df

    def get_feature_names_out(self, input_features=None):
        features = (
            list(input_features)
            if input_features is not None
            else (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else []
            )
        )
        return np.array(features, dtype=object)


class CabinIndicatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_original=True):
        """
        Args:
            drop_original (bool): If True, drops the raw 'Cabin' column
                                 after creating the indicator.
        """
        self.drop_original = drop_original
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = pd.DataFrame(X).columns.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df["HasCabin"] = X_df["Cabin"].notna().astype(int)

        if self.drop_original:
            X_df = X_df.drop(columns=["Cabin"])

        return X_df

    def get_feature_names_out(self, input_features=None):
        features = (
            list(input_features)
            if input_features is not None
            else (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else []
            )
        )
        if "HasCabin" not in features:
            features.append("HasCabin")

        if self.drop_original and "Cabin" in features:
            features.remove("Cabin")

        return np.array(features, dtype=object)


class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.most_frequent_embarked_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.most_frequent_embarked_ = X_df["Embarked"].mode()[0]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df["Embarked"] = X_df["Embarked"].fillna(self.most_frequent_embarked_)
        return X_df

    def get_feature_names_out(self, input_features=None):
        features = (
            list(input_features)
            if input_features is not None
            else (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else []
            )
        )
        return np.array(features, dtype=object)


class AutoSkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.skewed_cols_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()
        self.skewed_cols_ = []

        for col in X_df.select_dtypes(include=[np.number]).columns:
            # Skip columns that are binary (only contain 0, 1, or NaN)
            unique_vals = X_df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                continue

            skew_val = X_df[col].dropna().skew()
            if abs(skew_val) >= self.threshold:  # type: ignore
                self.skewed_cols_.append(col)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        for col in self.skewed_cols_:
            X_df[col] = np.log1p(X_df[col])

        return X_df

    def get_feature_names_out(self, input_features=None):
        features = (
            list(input_features)
            if input_features is not None
            else (
                list(self.feature_names_in_)
                if self.feature_names_in_ is not None
                else []
            )
        )
        return np.array(features, dtype=object)
