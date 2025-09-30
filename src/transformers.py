import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class FamilyBinExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        family_size = X.iloc[:, 0] + X.iloc[:, 1] + 1
        bins = [0, 1, 4, np.inf]
        labels = ['Single', 'Small', 'Large']
        family_bin = pd.cut(family_size, bins=bins, labels=labels)
        return family_bin.to_frame(name='FamilyBin')

    def get_feature_names_out(self, input_features=None):
        return ['FamilyBin']

class CabinExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        has_cabin = X.iloc[:, 0].notna().astype(int)
        return has_cabin.to_frame(name='HasCabin')

    def get_feature_names_out(self, input_features=None):
        return ['HasCabin']

class TitleBasedAgeImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that first engineers a 'Title' feature from the 'Name'
    column, and then imputes missing 'Age' values based on the median age
    for each title group.
    """

    def __init__(self, 
                 name_col='Name', 
                 age_col='Age', 
                 title_col='Title',
                 title_thres='Master'):
        self.name_col = name_col
        self.age_col = age_col
        self.title_col = title_col
        self.title_thres = title_thres

    def _clean_title(self, title):
        if title in ['Ms', 'Mlle']: return 'Miss'
        if title in ['Mme']: return 'Mrs'
        return title

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        X_fit = X.copy()
        titles = (
            X_fit[self.name_col]
            .str.extract(r' ([A-Za-z]+)\.', expand=False)
            .apply(self._clean_title)
        )
        counts = titles.value_counts()
        threshold = counts.get(self.title_thres, 0)
        self.rare_titles_ = counts[counts < threshold].index.tolist()
        X_fit[self.title_col] = titles.replace(self.rare_titles_, 'Rare')
        self.title_median_map_ = X_fit.groupby(self.title_col)[self.age_col].median().to_dict()
        self.global_median_ = X[self.age_col].median()

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ['feature_names_in_', 'rare_titles_', 'title_median_map_', 'global_median_']
        )

        if isinstance(X, np.ndarray):
            X_transformed = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_transformed = X.copy()

        titles = (
            X_transformed[self.name_col]
            .str.extract(r' ([A-Za-z]+)\.', expand=False)
            .apply(self._clean_title)
        )
        X_transformed[self.title_col] = titles.replace(self.rare_titles_, 'Rare')
        impute_values = X_transformed[self.title_col].map(self.title_median_map_)
        impute_values = impute_values.fillna(self.global_median_)
        X_transformed[self.age_col] = X_transformed[self.age_col].fillna(impute_values)

        return X_transformed[[self.age_col, self.title_col]]

    def get_feature_names_out(self, input_features=None):
        return [self.age_col, self.title_col]

class FareImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_copy = X.copy()
        self.median_fare_ = X_copy['Fare'].median()

        return self

    def transform(self, X):
        check_is_fitted(self, 'median_fare_')

        X_copy = X.copy()
        X_copy['Fare'] = X_copy['Fare'].fillna(self.median_fare_)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return ['Fare']

class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_copy = X.copy()
        self.mode_embarked_ = X_copy['Embarked'].mode()[0]

        return self

    def transform(self, X):
        check_is_fitted(self, 'mode_embarked_')
        X_copy = X.copy()
        X_copy['Embarked'] = X_copy['Embarked'].fillna(self.mode_embarked_)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return ['Embarked']