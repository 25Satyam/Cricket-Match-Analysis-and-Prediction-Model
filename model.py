"""
model.py — Cricket Match Prediction Models
==========================================
Trains two models:
  1. Score Predictor    – Random Forest Regressor  → predicts final innings score
  2. Win Probability    – Random Forest Classifier → P(chasing team wins) at any over
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class CricketPredictor:
    """Encapsulates training, prediction and simulation for cricket matches."""

    SCORE_FEATURES = [
        "over",
        "current_runs",
        "current_wickets",
        "overs_remaining",
        "current_rr",
        "wickets_remaining",
        "balls_faced",
    ]

    WIN_FEATURES = [
        "over",
        "target",
        "current_runs",
        "current_wickets",
        "runs_needed",
        "overs_remaining",
        "balls_remaining",
        "required_rr",
        "current_rr",
        "rr_diff",
        "wickets_remaining",
    ]

    def __init__(self):
        self.score_model = RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        self.win_model = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        self.scaler_score = StandardScaler()
        self.scaler_win = StandardScaler()
        self.is_trained: bool = False
        self.score_mae: float | None = None
        self.win_accuracy: float | None = None
        self.total_overs: int = 20

    # ─────────────────────────────────────────────────────────────────────────
    # Data generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_score_data(self, n_matches: int = 2500) -> pd.DataFrame:
        """Synthetic over-by-over first-innings data for score model."""
        rng = np.random.default_rng(42)
        total = self.total_overs
        records: list[dict] = []

        for _ in range(n_matches):
            # Realistic final score distributions
            if total == 20:
                final_score = int(rng.normal(165, 30))
                final_score = np.clip(final_score, 80, 280)
            else:
                final_score = int(rng.normal(280, 55))
                final_score = np.clip(final_score, 120, 480)

            wickets_fell = int(rng.integers(2, 11))

            # Build monotone runs progression
            run_steps = np.sort(rng.integers(0, final_score + 1, size=total - 1))
            run_steps = np.concatenate([[0], run_steps, [final_score]])

            # Distribute wickets across overs
            wicket_overs = sorted(
                rng.choice(range(1, total + 1), size=min(wickets_fell, total), replace=False)
            )
            wkt_prog = np.zeros(total + 1, dtype=int)
            for wo in wicket_overs:
                wkt_prog[wo:] += 1

            for ov in range(1, total + 1):
                cur_runs = int(run_steps[ov])
                cur_wkts = int(wkt_prog[ov])
                overs_rem = total - ov
                crr = cur_runs / ov
                records.append(
                    {
                        "over": ov,
                        "current_runs": cur_runs,
                        "current_wickets": cur_wkts,
                        "overs_remaining": overs_rem,
                        "current_rr": round(crr, 3),
                        "wickets_remaining": 10 - cur_wkts,
                        "balls_faced": ov * 6,
                        "final_score": final_score,
                    }
                )

        return pd.DataFrame(records)

    def _generate_win_data(self, n_matches: int = 2500) -> pd.DataFrame:
        """Synthetic over-by-over chase data for win-probability model."""
        rng = np.random.default_rng(99)
        total = self.total_overs
        records: list[dict] = []

        for _ in range(n_matches):
            if total == 20:
                target = int(rng.normal(165, 30))
                target = np.clip(target, 90, 280)
            else:
                target = int(rng.normal(280, 55))
                target = np.clip(target, 130, 480)

            chasing_wins = bool(rng.random() > 0.5)

            if chasing_wins:
                chase_total = target + int(rng.integers(1, 40))
            else:
                chase_total = target - int(rng.integers(5, 80))
            chase_total = int(np.clip(chase_total, 0, target + 60))

            cur_runs = 0
            cur_wkts = 0

            for ov in range(1, total + 1):
                progress = ov / total
                expected = int(chase_total * progress)
                noise = int(rng.normal(0, chase_total * 0.06))
                cur_runs = int(np.clip(cur_runs + max(0, (expected - cur_runs) + noise), 0, target + 60))
                if rng.random() < 0.13:
                    cur_wkts = min(10, cur_wkts + 1)

                overs_rem = total - ov
                balls_rem = overs_rem * 6
                runs_needed = max(0, target - cur_runs)
                req_rr = runs_needed / overs_rem if overs_rem > 0 else 99.0
                crr = cur_runs / ov
                records.append(
                    {
                        "over": ov,
                        "target": target,
                        "current_runs": cur_runs,
                        "current_wickets": cur_wkts,
                        "runs_needed": runs_needed,
                        "overs_remaining": overs_rem,
                        "balls_remaining": balls_rem,
                        "required_rr": round(min(req_rr, 36.0), 3),
                        "current_rr": round(crr, 3),
                        "rr_diff": round(crr - req_rr, 3),
                        "wickets_remaining": 10 - cur_wkts,
                        "win": int(chasing_wins),
                    }
                )

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, match_format: str = "T20") -> tuple[float, float]:
        """Train both models. Returns (score_mae, win_accuracy)."""
        self.total_overs = 20 if match_format == "T20" else 50

        # — Score model —
        score_df = self._generate_score_data()
        X_s = score_df[self.SCORE_FEATURES].values
        y_s = score_df["final_score"].values
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
        Xs_tr = self.scaler_score.fit_transform(Xs_tr)
        Xs_te = self.scaler_score.transform(Xs_te)
        self.score_model.fit(Xs_tr, ys_tr)
        self.score_mae = mean_absolute_error(ys_te, self.score_model.predict(Xs_te))

        # — Win model —
        win_df = self._generate_win_data()
        X_w = win_df[self.WIN_FEATURES].values
        y_w = win_df["win"].values
        Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
        Xw_tr = self.scaler_win.fit_transform(Xw_tr)
        Xw_te = self.scaler_win.transform(Xw_te)
        self.win_model.fit(Xw_tr, yw_tr)
        self.win_accuracy = accuracy_score(yw_te, self.win_model.predict(Xw_te))

        self.is_trained = True
        return self.score_mae, self.win_accuracy

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict_score(
        self,
        over: int,
        current_runs: int,
        current_wickets: int,
        total_overs: int | None = None,
    ) -> tuple[int, int, int]:
        """Return (predicted, lower_bound, upper_bound) final score."""
        if not self.is_trained:
            self.train()
        total = total_overs or self.total_overs
        features = np.array(
            [[
                over,
                current_runs,
                current_wickets,
                total - over,
                current_runs / over if over else 0,
                10 - current_wickets,
                over * 6,
            ]]
        )
        pred = self.score_model.predict(self.scaler_score.transform(features))[0]
        return int(pred), int(pred * 0.92), int(pred * 1.08)

    def predict_win_probability(
        self,
        target: int,
        current_runs: int,
        over: int,
        current_wickets: int,
        total_overs: int | None = None,
    ) -> float:
        """Return win probability (0–100) for the chasing team."""
        if not self.is_trained:
            self.train()
        total = total_overs or self.total_overs
        overs_rem = total - over
        balls_rem = overs_rem * 6
        runs_needed = max(0, target - current_runs)
        req_rr = runs_needed / overs_rem if overs_rem > 0 else 99.0
        crr = current_runs / over if over else 0.0
        features = np.array(
            [[
                over,
                target,
                current_runs,
                current_wickets,
                runs_needed,
                overs_rem,
                balls_rem,
                min(req_rr, 36.0),
                crr,
                crr - req_rr,
                10 - current_wickets,
            ]]
        )
        prob = self.win_model.predict_proba(self.scaler_win.transform(features))[0][1]
        return round(prob * 100, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_over_by_over(
        self,
        team1_score: int,
        total_overs: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate a full chase over-by-over and return a probability DataFrame."""
        if not self.is_trained:
            self.train()
        total = total_overs or self.total_overs
        target = team1_score + 1
        rng = np.random.default_rng(seed)

        records: list[dict] = []
        cur_runs = 0
        cur_wkts = 0

        for ov in range(1, total + 1):
            # Runs this over: scaled to target/total + noise
            base = target / total
            runs_this_over = max(0, int(rng.normal(base, base * 0.6)))
            if cur_wkts < 10:
                cur_runs += runs_this_over
                if rng.random() < 0.14:
                    cur_wkts = min(10, cur_wkts + 1)

            wp = self.predict_win_probability(target, cur_runs, ov, cur_wkts, total)
            records.append(
                {
                    "over": ov,
                    "runs": cur_runs,
                    "wickets": cur_wkts,
                    "win_probability": wp,
                    "lose_probability": round(100 - wp, 1),
                }
            )

        return pd.DataFrame(records)
