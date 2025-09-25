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

