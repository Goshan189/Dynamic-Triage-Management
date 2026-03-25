"""
ml_triage_model.py
------------------
ML triage pipeline.
Trains XGBoost and Random Forest on data.csv.
Risk scoring: cesd + stai_t + mbi_ex < 40 → Low,  < 70 → Medium,  >= 70 → High
Selects best model by accuracy.
Output: risk_label and priority_score (max predict_proba * 100).

Do NOT modify this file.  Use ml_bridge.py to integrate.
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Locate data.csv relative to this file's directory hierarchy.
# Accepted locations: same dir, parent, parent/parent.
# ──────────────────────────────────────────────────────────────────────────────
def _find_data_csv() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "data.csv"),
        os.path.join(base, "..", "data.csv"),
        os.path.join(base, "..", "..", "data.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "data.csv not found. Expected it next to ml_triage_model.py or in the "
        "hospital_routing parent directory."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Label creation
# ──────────────────────────────────────────────────────────────────────────────
def _assign_risk(row: pd.Series) -> str:
    score = row["cesd"] + row["stai_t"] + row["mbi_ex"]
    if score < 40:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "High Risk"


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────
class TriageModel:
    """Encapsulates the full training / prediction pipeline."""

    # Features used for training
    FEATURE_COLS = [
        "age", "year", "sex", "glang", "part", "job", "stud_h",
        "health", "psyt", "jspe", "qcae_cog", "qcae_aff", "amsp",
        "erec_mean", "cesd", "stai_t", "mbi_ex", "mbi_cy", "mbi_ea",
    ]

    def __init__(self) -> None:
        self.model          = None
        self.label_encoder  = LabelEncoder()
        self.feature_cols   = self.FEATURE_COLS
        self.best_model_name: str = ""
        self.accuracy: float = 0.0
        self._df: pd.DataFrame = pd.DataFrame()

    def load_and_train(self, data_path: str | None = None) -> None:
        if data_path is None:
            data_path = _find_data_csv()

        df = pd.read_csv(data_path)
        df = df.dropna(subset=["cesd", "stai_t", "mbi_ex"])
        df["risk_label"] = df.apply(_assign_risk, axis=1)
        self._df = df

        X = df[self.feature_cols].fillna(df[self.feature_cols].mean())
        y = self.label_encoder.fit_transform(df["risk_label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))

        # Select best
        if xgb_acc >= rf_acc:
            self.model           = xgb
            self.best_model_name = "XGBoost"
            self.accuracy        = xgb_acc
        else:
            self.model           = rf
            self.best_model_name = "RandomForest"
            self.accuracy        = rf_acc

    def predict(self, feature_row: dict) -> dict:
        """
        Predict risk for a single patient.

        Args:
            feature_row: dict with keys matching FEATURE_COLS.
                         Missing keys default to column mean from training data.

        Returns:
            {
                "risk_label":     "High Risk" | "Medium Risk" | "Low Risk",
                "priority_score": float 0–100,
                "model_used":     str,
            }
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call load_and_train() first.")

        # Build feature vector
        row_vals = []
        col_means = (
            self._df[self.feature_cols].mean()
            if not self._df.empty
            else {c: 0.0 for c in self.feature_cols}
        )
        for col in self.feature_cols:
            val = feature_row.get(col, col_means[col])
            row_vals.append(float(val))

        X_pred = np.array([row_vals])
        proba  = self.model.predict_proba(X_pred)[0]   # shape (n_classes,)
        class_idx = int(np.argmax(proba))
        risk_label = self.label_encoder.inverse_transform([class_idx])[0]
        priority_score = float(np.max(proba)) * 100.0

        return {
            "risk_label":     risk_label,
            "priority_score": priority_score,
            "model_used":     self.best_model_name,
        }

    def get_column_stats(self) -> dict:
        """
        Return {column: {"mean": float, "std": float}} for synthetic vitals
        generation in ml_bridge.py.
        """
        if self._df.empty:
            raise RuntimeError("Model not trained.")
        stats = {}
        for col in self.feature_cols:
            stats[col] = {
                "mean": float(self._df[col].mean()),
                "std":  float(self._df[col].std()),
            }
        return stats
