"""
generate_data.py
================
Generates synthetic cricket match datasets and saves them to data/.
Run once to create the sample CSV files used for exploratory analysis.

Usage:
    python generate_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

RNG = np.random.default_rng(2024)

TEAMS = [
    "India", "Australia", "England", "New Zealand",
    "South Africa", "Pakistan", "West Indies", "Sri Lanka",
    "Bangladesh", "Afghanistan",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "Eden Gardens, Kolkata",
    "MCG, Melbourne",
    "Lord's, London",
    "Newlands, Cape Town",
    "Gaddafi Stadium, Lahore",
    "Sabina Park, Kingston",
    "R. Premadasa Stadium, Colombo",
    "Shere Bangla, Dhaka",
    "Sharjah Cricket Stadium",
]


def simulate_innings(total_overs: int = 20) -> tuple[list[int], list[int], int]:
    """Return runs per over, wickets per over, and final score."""
    if total_overs == 20:
        final = int(np.clip(RNG.normal(165, 28), 80, 280))
    else:
        final = int(np.clip(RNG.normal(275, 52), 120, 480))

    # Runs distribution — powerplay higher, death higher, middle moderate
    weights = []
    for ov in range(1, total_overs + 1):
        p = ov / total_overs
        if p <= 0.30:
            weights.append(1.20)
        elif p <= 0.70:
            weights.append(0.90)
        else:
            weights.append(1.30)
    weights = np.array(weights)
    weights = weights / weights.sum()

    runs_per_over = RNG.multinomial(final, weights).tolist()

    # Wickets
    wickets_per_over = [0] * total_overs
    total_wickets = int(RNG.integers(2, 11))
    wkt_overs = RNG.choice(range(total_overs), size=min(total_wickets, total_overs), replace=False)
    for wo in wkt_overs:
        wickets_per_over[wo] += 1

    return runs_per_over, wickets_per_over, final


def generate_matches(n: int = 500, match_format: str = "T20") -> pd.DataFrame:
    total_overs = 20 if match_format == "T20" else 50
    records = []

    for match_id in range(1, n + 1):
        team_pair = RNG.choice(len(TEAMS), size=2, replace=False)
        t1 = TEAMS[team_pair[0]]
        t2 = TEAMS[team_pair[1]]
        venue = VENUES[RNG.integers(len(VENUES))]
        toss_winner = RNG.choice([t1, t2])
        toss_decision = RNG.choice(["bat", "field"])

        # Determine which team bats first
        batting_first = toss_winner if toss_decision == "bat" else (t2 if toss_winner == t1 else t1)
        batting_second = t2 if batting_first == t1 else t1

        # Innings 1
        runs1, wkts1, total1 = simulate_innings(total_overs)
        cum_runs = 0
        cum_wkts = 0
        for ov_idx, (r, w) in enumerate(zip(runs1, wkts1)):
            ov = ov_idx + 1
            cum_runs += r
            cum_wkts = min(10, cum_wkts + w)
            records.append(
                {
                    "match_id": match_id,
                    "format": match_format,
                    "venue": venue,
                    "team": batting_first,
                    "opponent": batting_second,
                    "innings": 1,
                    "over": ov,
                    "runs_this_over": r,
                    "wickets_this_over": w,
                    "cumulative_runs": cum_runs,
                    "cumulative_wickets": cum_wkts,
                    "final_innings_score": total1,
                    "target": None,
                    "result": None,
                }
            )

        # Innings 2 (chase)
        target = total1 + 1
        runs2, wkts2, _ = simulate_innings(total_overs)
        chase_total = sum(runs2)
        result = batting_second if chase_total >= total1 else batting_first

        cum_runs = 0
        cum_wkts = 0
        completed = False
        for ov_idx, (r, w) in enumerate(zip(runs2, wkts2)):
            if completed:
                break
            ov = ov_idx + 1
            cum_runs += r
            cum_wkts = min(10, cum_wkts + w)
            if cum_runs >= target:
                completed = True
            records.append(
                {
                    "match_id": match_id,
                    "format": match_format,
                    "venue": venue,
                    "team": batting_second,
                    "opponent": batting_first,
                    "innings": 2,
                    "over": ov,
                    "runs_this_over": r,
                    "wickets_this_over": w,
                    "cumulative_runs": cum_runs,
                    "cumulative_wickets": cum_wkts,
                    "final_innings_score": chase_total,
                    "target": target,
                    "result": result,
                }
            )

    return pd.DataFrame(records)


def main():
    print("Generating T20 match data …")
    t20_df = generate_matches(400, "T20")
    t20_path = os.path.join(DATA_DIR, "t20_matches.csv")
    t20_df.to_csv(t20_path, index=False)
    print(f"  Saved {len(t20_df):,} rows → {t20_path}")

    print("Generating ODI match data …")
    odi_df = generate_matches(200, "ODI")
    odi_path = os.path.join(DATA_DIR, "odi_matches.csv")
    odi_df.to_csv(odi_path, index=False)
    print(f"  Saved {len(odi_df):,} rows → {odi_path}")

    # Summary stats
    print("\nT20 summary:")
    t20_summary = (
        t20_df[t20_df["innings"] == 1]
        .groupby("match_id")["final_innings_score"]
        .max()
        .describe()
    )
    print(t20_summary.to_string())

    print("\nDone! Files saved to ./data/")


if __name__ == "__main__":
    main()
