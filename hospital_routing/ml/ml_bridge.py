"""
ml_bridge.py
------------
Integration bridge between the ML triage pipeline and the rest of the system.

Public API:
    MLBridge.classify(patient_vitals: dict) -> dict
        Input:  feature row matching data.csv columns
        Output: {
            "priority":       "high" | "medium" | "low",
            "priority_score": float (0–100),
            "risk_label":     "High Risk" | "Medium Risk" | "Low Risk",
        }

    MLBridge.generate_synthetic_vitals() -> dict
        Samples one synthetic feature row from column distributions in data.csv.
        Used by the simulation engine.
"""

from __future__ import annotations

import numpy as np

from ml.ml_triage_model import TriageModel

_RISK_TO_PRIORITY = {
    "High Risk":   "high",
    "Medium Risk": "medium",
    "Low Risk":    "low",
}


class MLBridge:
    """Singleton-style wrapper around TriageModel."""

    def __init__(self, data_path: str | None = None) -> None:
        self._model = TriageModel()
        self._model.load_and_train(data_path)
        self._col_stats = self._model.get_column_stats()
        self._rng = np.random.default_rng(seed=None)

    # ── classification ────────────────────────────────────────────────────────
    def classify(self, patient_vitals: dict) -> dict:
        """
        Run the ML pipeline on a feature dict.

        Returns:
            {
                "priority":       "high" | "medium" | "low",
                "priority_score": float 0–100,
                "risk_label":     "High Risk" | "Medium Risk" | "Low Risk",
            }
        """
        result = self._model.predict(patient_vitals)
        risk_label     = result["risk_label"]
        priority_score = result["priority_score"]
        priority       = _RISK_TO_PRIORITY[risk_label]

        return {
            "priority":       priority,
            "priority_score": priority_score,
            "risk_label":     risk_label,
        }

    # ── synthetic vitals ──────────────────────────────────────────────────────
    def generate_synthetic_vitals(self) -> dict:
        """
        Sample a synthetic patient vitals row from the training distribution.
        Uses mean ± std per column (Gaussian), clipped to observed range where
        appropriate (no negative values for count-like columns).
        """
        vitals = {}
        for col, stats in self._col_stats.items():
            mean = stats["mean"]
            std  = stats["std"]
            val  = float(self._rng.normal(mean, std))
            # Clip non-negative columns
            if col in ("age", "stud_h", "cesd", "stai_t", "mbi_ex",
                       "mbi_cy", "mbi_ea", "jspe", "qcae_cog", "qcae_aff",
                       "amsp", "year"):
                val = max(0.0, val)
            vitals[col] = val
        return vitals

    # ── model info ────────────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return self._model.best_model_name

    @property
    def accuracy(self) -> float:
        return self._model.accuracy
