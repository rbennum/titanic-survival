import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FamilySizeBinner(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 sibling_col='SibSp',
                 parent_col='Parch',
                 output_bin_col='FamilyBin',
                 bins=[0, 1, 4, np.inf],
                 labels=['small', 'medium', 'large'],
                 drop_originals=False):
        self.sibling_col = sibling_col
        self.parent_col = parent_col
        self.output_bin_col = output_bin_col
        self.drop_originals = drop_originals
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_temp = X.copy()

        family_size = df_temp[self.sibling_col] + df_temp[self.parent_col] + 1

        df_temp[self.output_bin_col] = pd.cut(
            family_size, 
            bins=self.bins,
            labels=self.labels
        )

        if self.drop_originals:
            df_temp = df_temp.drop(columns=[self.sibling_col, self.parent_col])

        return df_temp

class TitleFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to engineer a 'Title' feature from the 'Name' column.
    
    It extracts, cleans, and consolidates titles, grouping rare titles into a
    single 'Rare' category based on the training data distribution.
    """
    def __init__(self, name_col='Name', title_col='Title', rare_threshold_ref='Master'):
        self.name_col = name_col
        self.title_col = title_col
        self.rare_threshold_ref = rare_threshold_ref

    def _clean_title(self, title):
        if title in ['Ms', 'Mlle']:
            return 'Miss'
        if title in ['Mme', 'Lady']:
            return 'Mrs'
        return title

    def fit(self, X, y=None):
        df_temp = X.copy()
        titles = (
            df_temp[self.name_col]
            .str.extract(r' ([A-Za-z]+)\.', expand=False)
            .apply(self._clean_title)
        )
        counts = titles.value_counts()
        threshold = counts.get(self.rare_threshold_ref, 0)
        self.rare_titles_ = counts[counts < threshold].index.tolist()

        return self

    def transform(self, X):
        df_temp = X.copy()
        titles = (
            df_temp[self.name_col]
            .str.extract(r' ([A-Za-z]+)\.', expand=False)
            .apply(self._clean_title)
        )
        df_temp[self.title_col] = titles.replace(self.rare_titles_, 'Rare')

        return df_temp