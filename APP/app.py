"""
Smart AI-Based Loan Decision Support System
Backend: Flask + scikit-learn inference pipeline
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "loan_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "feature_columns.pkl")
model = None
feature_columns = None

def load_artifacts():
    global model, feature_columns
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            feature_columns = pickle.load(f)
        print("[INFO] Model and features loaded successfully.")
    else:
        print("[WARN] Model files not found – running in demo/mock mode.")

load_artifacts()

# ─────────────────────────────────────────────
# 2. INPUT SCHEMA  (all 32 raw dataset fields)
# ─────────────────────────────────────────────

REQUIRED_FIELDS = [
    "year", "loan_limit", "Gender", "approv_in_adv", "loan_type",
    "loan_purpose", "Credit_Worthiness", "open_credit",
    "business_or_commercial", "loan_amount", "rate_of_interest",
    "Interest_rate_spread", "Upfront_charges", "term",
    "Neg_ammortization", "interest_only", "lump_sum_payment",
    "property_value", "construction_type", "occupancy_type",
    "Secured_by", "total_units", "income", "credit_type",
    "Credit_Score", "co-applicant_credit_type", "age",
    "submission_of_application", "LTV", "Region",
    "Security_Type", "dtir1",
]

NUMERIC_FIELDS = [
    "year", "loan_amount", "rate_of_interest", "Interest_rate_spread",
    "Upfront_charges", "term", "property_value", 
    "income", "Credit_Score", "LTV", "dtir1",
]

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(raw: dict) -> pd.DataFrame:
    """
    Convert raw dict → one-row DataFrame with get_dummies + feature alignment.
    """
    # Cast numeric fields
    for field in NUMERIC_FIELDS:
        if field in raw:
            raw[field] = float(raw[field])

    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, drop_first=True)

    # Align to saved feature columns
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    return df

# ─────────────────────────────────────────────
# 4. FINANCE HELPERS
# ─────────────────────────────────────────────

def compute_emi(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Standard EMI formula:  P * r * (1+r)^n / ((1+r)^n - 1)
    """
    if annual_rate <= 0 or term_months <= 0:
        return principal / max(term_months, 1)
    r = annual_rate / 100 / 12
    n = term_months
    emi = principal * r * math.pow(1 + r, n) / (math.pow(1 + r, n) - 1)
    return round(emi, 2)


def emi_to_income_ratio(emi: float, monthly_income: float) -> float:
    if monthly_income <= 0:
        return 999.0
    return round((emi / monthly_income) * 100, 2)


def safe_loan_amount(monthly_income: float, annual_rate: float,
                     term_months: int, max_ratio: float = 0.35) -> float:
    """
    Back-solve: max affordable loan given that EMI ≤ max_ratio * income.
    Uses standard annuity present-value formula.
    """
    max_emi = monthly_income * max_ratio
    if annual_rate <= 0 or term_months <= 0:
        return round(max_emi * term_months, 2)
    r = annual_rate / 100 / 12
    n = term_months
    # P = EMI * [(1+r)^n - 1] / [r * (1+r)^n]
    safe_principal = max_emi * (math.pow(1 + r, n) - 1) / (r * math.pow(1 + r, n))
    return round(safe_principal, 2)

# ─────────────────────────────────────────────
# 5. RISK + DECISION RULES
# ─────────────────────────────────────────────

def probability_to_risk(prob: float) -> str:
    if prob < 0.35:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    return "High Risk"


def make_decision(risk: str, emi_ratio: float) -> str:
    if risk == "Low Risk":
        return "Approve"
    elif risk == "High Risk":
        return "Reject"
    else:  # Medium
        if emi_ratio > 50:
            return "Request Lower Loan / Additional Verification"
        return "Approve with Conditions"

# ─────────────────────────────────────────────
# 6. EXPLANATION ENGINE
# ─────────────────────────────────────────────

def generate_reasons(raw: dict, emi: float, emi_ratio: float,
                     default_prob: float) -> list[str]:
    reasons = []
    income      = float(raw.get("income", 0))
    loan_amount = float(raw.get("loan_amount", 0))
    credit      = float(raw.get("Credit_Score", 0))
    ltv         = float(raw.get("LTV", 0))
    dtir        = float(raw.get("dtir1", 0))

    if income < 3000:
        reasons.append("Low monthly income increases repayment risk.")
    if loan_amount > 5 * income * 12:
        reasons.append("Loan amount is very high relative to annual income.")
    if credit < 600:
        reasons.append("Low credit score signals higher probability of default.")
    elif credit < 680:
        reasons.append("Below-average credit score adds moderate risk.")
    if dtir > 43:
        reasons.append("High debt-to-income ratio (DTI) exceeds safe threshold of 43%.")
    if ltv > 80:
        reasons.append("LTV above 80% indicates insufficient equity cushion.")
    if emi_ratio > 50:
        reasons.append(f"EMI-to-income ratio of {emi_ratio}% is dangerously high (safe ≤ 35%).")
    elif emi_ratio > 35:
        reasons.append(f"EMI-to-income ratio of {emi_ratio}% is elevated (recommended ≤ 35%).")
    if default_prob >= 0.70:
        reasons.append("Model assigns very high probability of default based on applicant profile.")
    elif default_prob >= 0.35:
        reasons.append("Moderate default probability detected – conditional approval warranted.")

    if not reasons:
        reasons.append("Applicant profile appears financially sound with manageable risk.")
    return reasons

# ─────────────────────────────────────────────
# 7. MOCK PREDICTION (when model files absent)
# ─────────────────────────────────────────────

def mock_predict(raw: dict) -> tuple[int, float]:
    """
    Deterministic heuristic mock for demo / development without .pkl files.
    Returns (label, default_probability).
    """
    credit = float(raw.get("Credit_Score", 650))
    dtir   = float(raw.get("dtir1", 30))
    ltv    = float(raw.get("LTV", 70))

    score = 0
    if credit < 580:  score += 40
    elif credit < 650: score += 20
    elif credit > 750: score -= 20

    if dtir > 50:  score += 30
    elif dtir > 40: score += 15

    if ltv > 90:  score += 25
    elif ltv > 80: score += 10

    prob = min(max(score / 100, 0.05), 0.95)
    label = 1 if prob >= 0.5 else 0
    return label, round(prob, 4)

# ─────────────────────────────────────────────
# 8. ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw = request.get_json(force=True)
        if not raw:
            return jsonify({"error": "No JSON body received."}), 400

        # --- Validate required fields ---
        missing = [f for f in REQUIRED_FIELDS if f not in raw or raw[f] == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 422

        # --- Finance pre-computations ---
        loan_amount    = float(raw["loan_amount"])
        annual_rate    = float(raw["rate_of_interest"])
        term_months    = int(float(raw["term"]))
        monthly_income = float(raw["income"])

        emi       = compute_emi(loan_amount, annual_rate, term_months)
        emi_ratio = emi_to_income_ratio(emi, monthly_income)

        # --- Prediction ---
        if model is not None and feature_columns is not None:
            df      = preprocess(dict(raw))
            prob    = float(model.predict_proba(df)[0][1])
            label   = int(model.predict(df)[0])
        else:
            label, prob = mock_predict(raw)

        # --- Risk + decision ---
        risk     = probability_to_risk(prob)
        decision = make_decision(risk, emi_ratio)
        outcome  = "Default" if label == 1 else "No Default"

        # --- Safe loan ---
        safe_amount = safe_loan_amount(monthly_income, annual_rate, term_months)

        # --- Explanations ---
        reasons = generate_reasons(raw, emi, emi_ratio, prob)

        return jsonify({
            "outcome":         outcome,
            "default_probability": round(prob * 100, 2),
            "risk_level":      risk,
            "decision":        decision,
            "emi":             emi,
            "emi_ratio":       emi_ratio,
            "safe_loan_amount": safe_amount,
            "reasons":         reasons,
            "mock_mode":       model is None,
        })

    except ValueError as ve:
        return jsonify({"error": f"Value error: {str(ve)}"}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))