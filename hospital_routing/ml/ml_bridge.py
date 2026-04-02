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

import logging
import random
from typing import Dict, Optional, List

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

    def get_clinical_allocation(self, vitals: dict, priority: str) -> tuple[List[str], str]:
        """
        Maps a patient's exact psychological scores to a logical hospital department,
        and provides the medical reasoning string.
        """
        from graph.layout import PRIORITY_DESTINATIONS
        import random
        
        valid_dests = PRIORITY_DESTINATIONS[priority]
        reason = ""
        preferred = random.choice(valid_dests) # fallback

        cesd = vitals.get("cesd", 0.0)
        stai = vitals.get("stai_t", 0.0)
        mbi  = vitals.get("mbi_ex", 0.0)
        health = vitals.get("health", 5.0)

        if priority == "high":
            # Wards get first grab at high-anxiety patients
            if stai > 50:
                preferred = ["ward_surgical", "icu"]
                reason = f"Severe anxiety (STAI {stai:.0f}). Routing to quiet Surgical Ward or ICU for isolated monitoring."
            elif cesd > 45:
                preferred = ["icu"]
                reason = f"Critical depression (CES-D {cesd:.0f}). Requires ICU psychiatric stabilization."
            elif mbi > 40:
                preferred = ["emergency", "icu"]
                reason = f"Severe exhaustion (MBI {mbi:.0f}). Routing to Emergency/ICU for acute burnout intervention."
            else:
                preferred = ["emergency", "icu", "ward_surgical"]
                reason = "High aggregate risk score. Fast-tracking to broad acute care."

        elif priority == "medium":
            # Wards get first grab at medium depression/burnout
            if cesd > 25:
                preferred = ["ward_general", "ward_surgical"]
                reason = f"Clinical depression (CES-D {cesd:.0f}). Routing to General/Surgical Ward for observation."
            elif mbi > 30:
                preferred = ["ward_surgical", "ward_general"]
                reason = f"Moderate exhaustion (MBI {mbi:.0f}). Routing to Wards for inpatient rest."
            elif stai > 45:
                preferred = ["lab", "radiology"]
                reason = f"Elevated anxiety (STAI {stai:.0f}). Routing to Diagnostic Wing for stress biomarker/screening panel."
            elif health < 3:
                preferred = ["radiology", "lab"]
                reason = f"Poor systemic health (Self-Score {health:.0f}). Routing to Diagnostics for physical screening."
            else:
                preferred = ["emergency", "ward_general"]
                reason = "Medium aggregate burnout risk. Routing to Emergency triage or General Ward."
                
        else: # low
            if mbi > 20:
                preferred = ["pharmacy"]
                reason = f"Mild exhaustion (MBI {mbi:.0f}). Routing to Pharmacy for outpatient supplements."
            elif stai > 35:
                preferred = ["opd_a", "opd_b"]
                reason = f"Mild anxiety (STAI {stai:.0f}). Routing to standard OPD clinics for routine counseling."
            else:
                preferred = ["opd_b", "opd_a", "pharmacy"]
                reason = "Standard baseline vitals. Routing to Outpatient clinics for general checkup."

        # Ensure preferred targets are physically accessible for this node priority
        valid_preferred = [p for p in preferred if p in valid_dests]
        if not valid_preferred:
            # Fallback to absolute load balancing across every valid destination assigned to this priority
            valid_preferred = valid_dests
            reason = f"{reason} (Fallback: Matrix Load Balancing constraint)"

        return valid_preferred, reason

    # ── synthetic vitals ──────────────────────────────────────────────────────
    def generate_synthetic_vitals(self, target_risk: str | None = None) -> dict:
        """
        Sample a realistic patient vitals row directly from the training dataset.
        If target_risk is specified (e.g. 'Low Risk'), we sample only from rows
        with that label, guaranteeing our desired statistical distributions.
        """
        df = self._model._df
        if df.empty:
            # Fallback if model isn't trained
            return self._fallback_synthetic_vitals()
            
        if target_risk:
            subset = df[df["risk_label"] == target_risk]
            if not subset.empty:
                idx = self._rng.integers(0, len(subset))
                row = subset.iloc[idx]
            else:
                idx = self._rng.integers(0, len(df))
                row = df.iloc[idx]
        else:
            idx = self._rng.integers(0, len(df))
            row = df.iloc[idx]
            
        vitals = {}
        for col in self._model.feature_cols:
            vitals[col] = float(row[col])
        return vitals

    def _fallback_synthetic_vitals(self) -> dict:
        """
        Old Gaussian estimation method.
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

    def get_patient_row(self, row_index: int) -> dict:
        """Fetch a specific row from data.csv for injection."""
        df = self._model._df
        if df.empty or row_index < 0 or row_index >= len(df):
            return self.generate_synthetic_vitals()
        
        row = df.iloc[row_index]
        vitals = {}
        for col in self._model.feature_cols:
            vitals[col] = float(row[col])
        return vitals

    # ── model info ────────────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return self._model.best_model_name

    @property
    def accuracy(self) -> float:
        return self._model.accuracy
