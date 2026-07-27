from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(title="Loan Default Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

# build the SHAP explainer once at startup, reused for every request
explainer = shap.TreeExplainer(model)


class Applicant(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30to59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60to89DaysPastDueNotWorse: int
    NumberOfDependents: int


@app.get("/")
def home():
    return {"message": "Loan Default Risk API is running"}


@app.post("/predict")
def predict(applicant: Applicant):
    total_past_due = (
        applicant.NumberOfTime30to59DaysPastDueNotWorse
        + applicant.NumberOfTime60to89DaysPastDueNotWorse
        + applicant.NumberOfTimes90DaysLate
    )
    income_per_dependent = applicant.MonthlyIncome / (applicant.NumberOfDependents + 1)

    row = {
        "RevolvingUtilizationOfUnsecuredLines": applicant.RevolvingUtilizationOfUnsecuredLines,
        "age": applicant.age,
        "NumberOfTime30-59DaysPastDueNotWorse": applicant.NumberOfTime30to59DaysPastDueNotWorse,
        "DebtRatio": applicant.DebtRatio,
        "MonthlyIncome": applicant.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": applicant.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": applicant.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": applicant.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": applicant.NumberOfTime60to89DaysPastDueNotWorse,
        "NumberOfDependents": applicant.NumberOfDependents,
        "TotalPastDue": total_past_due,
        "IncomePerDependent": income_per_dependent,
    }

    X = pd.DataFrame([row])[feature_names]

    probability = model.predict_proba(X)[0][1]

    if probability < 0.15:
        band = "Low"
    elif probability < 0.4:
        band = "Medium"
    else:
        band = "High"

    # SHAP explanation for this single applicant
    shap_out = explainer.shap_values(X)
    if isinstance(shap_out, list):
        vals = shap_out[1][0]
    elif hasattr(shap_out, "ndim") and shap_out.ndim == 3:
        vals = shap_out[0, :, 1]
    else:
        vals = shap_out[0]

    contributions = sorted(
        zip(feature_names, vals),
        key=lambda p: p[1],
        reverse=True,
    )
    top_factors = [
        {"feature": f, "impact": round(float(v), 4)}
        for f, v in contributions[:3]
        if v > 0
    ]

    return {
        "risk_probability": round(float(probability), 4),
        "risk_band": band,
        "top_factors": top_factors,
    }