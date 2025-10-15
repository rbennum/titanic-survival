import pandas as pd
import numpy as np
import time

def skim_data(data) -> pd.DataFrame:
    """
    Skims the dataframe for features, feature types, null, negative, and zero values
    percentage, and the number of unique values.

    :param DataFrame data: The input dataframe.
    :return: A dataframe contains the summary of the input dataframe.
    :rtype: DataFrame
    """

    numeric_cols = set(data.select_dtypes(include=[np.number]).columns)
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            'neg_%': round((data[col] < 0).mean() * 100, 3),
            'zero_%': round((data[col] == 0).mean() * 100, 3)
        }

    skimmed_data = pd.DataFrame({
        'feature': data.columns.values,
        'dtype': data.dtypes.astype(str).values,
        'null_%': round(data.isna().mean() * 100, 3).values,
        'negative_%': [numeric_stats.get(col, {}).get('neg_%', '-') for col in data.columns],
        'zero_%': [numeric_stats.get(col, {}).get('zero_%', '-') for col in data.columns],
        'n_unique': data.nunique().values,
        'unique_%': round(data.nunique() / len(data) * 100, 2).values,
        'sample_values': [list(data[col].dropna().unique()[:5]) for col in data.columns]
    })

    print(f'Total duplicate rows: {data.duplicated().sum()}')
    print(f'DF shape: {data.shape}')

    return skimmed_data

def create_submission(passenger_ids, predictions):
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    filename = f'input/submission_{time.time()}.csv'
    submission_df.to_csv(filename, index=False)
    print(f"\nSubmission file '{filename}' created successfully!")