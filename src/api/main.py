from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survivability API", 
              description="An API to serve Titanic survivability predictions")

model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

