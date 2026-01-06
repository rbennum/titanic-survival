import os
import joblib
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional

from preprocessor import TitanicPreprocessor
import logging

logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    preprocessor = TitanicPreprocessor(models_path=MODELS_DIR)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessor: {e}")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="An API to predict passenger survival on the Titanic.",
    version="1.0.0",
)


class Passenger(BaseModel):
    """Defines the input data structure for a single passenger."""

    pclass: Literal[1, 2, 3] = Field(..., description="Ticket class")
    sex: Literal["male", "female"] = Field(..., description="Sex")
    age: Optional[float] = Field(None, ge=0, le=90, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: Optional[Literal["S", "C", "Q"]] = Field(
        None, description="Port of Embarkation"
    )
    name: str = Field(
        "Olsen, Mr. Karl Siegwart",
        description="Passenger's full name (for Title extraction)",
    )
    ticket: str = Field(
        "11778", description="Ticket number (for Fare_Per_Person calculation)"
    )
    cabin: Optional[str] = Field(
        "B37", description="Cabin number (for Cabin_Letter extraction)"
    )


@app.get("/", tags=["General"])
def read_root():
    """A root endpoint to check if the API is running."""
    return {"message": "Welcome to the Titanic Survival Prediction API!"}


@app.post("/predict", tags=["Prediction"])
def predict_survival(passenger: Passenger):
    """
    Predicts survival for a single passenger.

    - **Receives**: A JSON object with passenger details.
    - **Performs**: Data preprocessing and model inference.
    - **Returns**: A JSON object with the survival prediction and probability.
    """
    try:
        input_df = pd.DataFrame([passenger.model_dump()])
        processed_df = preprocessor.transform(input_df)
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0]
        survival_probability = probability[1]

        return {
            "prediction": "Survived" if prediction == 1 else "Did not survive",
            "survived": bool(prediction == 1),
            "survival_probability": float(survival_probability),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )
