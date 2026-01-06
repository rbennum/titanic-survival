import pandas as pd
import joblib
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class TitanicPreprocessor:
    """
    A class to handle all preprocessing for the Titanic dataset.
    It loads necessary artifacts and applies a series of transformations.
    """

    def __init__(self, models_path: str):
        """
        Initializes the preprocessor by loading artifacts.
        :param models_path: Path to the directory containing saved model artifacts.
        """
        self.age_lookup = joblib.load(os.path.join(models_path, "age_lookup.joblib"))
        self.ticket_counts = joblib.load(
            os.path.join(models_path, "ticket_counts.joblib")
        )
        self.fare_lookup = joblib.load(os.path.join(models_path, "fare_lookup.joblib"))
        self.embarked_mode = joblib.load(
            os.path.join(models_path, "embarked_mode.joblib")
        )
        self.transformer = joblib.load(
            os.path.join(models_path, "data_transformer.joblib")
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the full preprocessing pipeline.
        """
        df_copy = df.copy()
        df_copy = self._remove_home_dest(df_copy)
        df_copy = self._apply_age_feature(df_copy, self.age_lookup)
        df_copy = self._apply_cabin_feature(df_copy)
        df_copy = self._apply_family_size_feature(df_copy)
        df_copy = self._apply_name_feature(df_copy)
        df_copy = self._apply_fare_feature(
            df_copy, self.ticket_counts, self.fare_lookup
        )
        df_copy = self._apply_embarked_feature(df_copy, self.embarked_mode)
        df_copy = self._apply_sex_feature(df_copy)
        logger.debug(df_copy.to_dict())
        df_copy = self._apply_ohe(df_copy, self.transformer)
        return df_copy

    def _remove_home_dest(self, df):
        df_copy = df.copy()
        # return df_copy.drop("home.dest", axis=1)
        return df_copy

    def _apply_cabin_feature(self, df):
        df_copy = df.copy()
        df_copy["cabin"] = df_copy["cabin"].fillna("M")
        df_copy["cabin"] = df_copy["cabin"].str[0]
        df_copy.loc[df_copy["cabin"] == "T", "cabin"] = "A"
        df_copy["cabin"] = (
            df_copy["cabin"]
            .replace(["A", "B", "C"], "ABC")
            .replace(["D", "E", "F", "G"], "DEFG")
        )
        df_copy["has_cabin"] = (df_copy["cabin"] != "M") * 1
        return df_copy

    def _apply_name_feature(self, df):
        df_temp = df.copy()
        df_temp["title"] = df_temp["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        df_temp["title"] = (
            df_temp["title"]
            .replace("Don", "Mr")
            .replace("Dona", "Mrs")
            .replace("Mlle", "Ms")
            .replace("Mme", "Mrs")
            .replace("Miss", "Ms")
        )
        df_temp["title"] = df_temp["title"].replace(
            [
                "Dr",
                "Jonkheer",
                "Rev",
                "Sir",
                "Lady",
                "Col",
                "Major",
                "Countess",
                "Capt",
                "Master",
            ],
            "Rare",
        )
        return df_temp.drop("name", axis=1)

    def _apply_age_feature(self, df, lookup):
        df_temp = df.copy()
        median = df_temp.set_index(["pclass", "sex"]).index.map(lookup)
        df_temp["age"] = df_temp["age"].fillna(pd.Series(median, index=df_temp.index))
        return df_temp

    def _apply_fare_feature(self, df, counts, lookup):
        df_copy = df.copy()
        median = df_copy.set_index(["pclass", "sex"]).index.map(lookup)
        df_copy["fare"] = df_copy["fare"].fillna(pd.Series(median, index=df_copy.index))
        df_copy["people_in_ticket"] = df_copy["ticket"].map(counts)
        df_copy["people_in_ticket"] = df_copy["people_in_ticket"].fillna(1)
        df_copy["fare_per_person"] = df_copy["fare"] / df_copy["people_in_ticket"]
        return df_copy.drop(columns=["people_in_ticket", "ticket", "fare"])

    def _apply_family_size_feature(self, df):
        df_temp = df.copy()
        df_temp["family_size"] = df["parch"] + df["sibsp"] + 1
        bins = [0, 1, 4, 20]
        labels = ["alone", "middle", "large"]
        df_temp["family_size"] = pd.cut(
            df_temp["family_size"], bins=bins, labels=labels
        )
        return df_temp.drop(columns=["parch", "sibsp"])

    def _apply_embarked_feature(self, df, mode):
        df_copy = df.copy()
        df_copy["embarked"] = df_copy["embarked"].fillna(mode)
        return df_copy

    def _apply_sex_feature(self, df):
        df_copy = df.copy()
        df_copy["sex"] = df_copy["sex"].map({"female": 0, "male": 1})
        return df_copy

    def _apply_ohe(self, df, transformer):
        return transformer.transform(df)
