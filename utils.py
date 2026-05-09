"""utils.py — Helper functions for Cricket Analysis app."""

from __future__ import annotations


def calculate_run_rate(runs: int | float, overs: int | float) -> float:
    """Current run rate = runs / overs completed."""
    if overs <= 0:
        return 0.0
    return round(runs / overs, 2)


def calculate_required_run_rate(runs_needed: int | float, overs_remaining: int | float) -> float:
    """Required run rate for the chasing team."""
    if overs_remaining <= 0:
        return 99.99
    return round(runs_needed / overs_remaining, 2)


def get_match_phase(over: int, total_overs: int = 20) -> str:
    """Return the current match phase label."""
    progress = over / total_overs
    if progress <= 0.30:
        return "⚡ Powerplay"
    elif progress <= 0.70:
        return "🔄 Middle Overs"
    else:
        return "💀 Death Overs"


def format_score(runs: int, wickets: int) -> str:
    """Return score in 'runs/wickets' format."""
    return f"{runs}/{wickets}"


def get_chase_difficulty(required_rr: float, match_format: str = "T20") -> str:
    """Classify how hard a chase is based on required run rate."""
    thresholds: dict[str, list[tuple[float, str]]] = {
        "T20": [
            (6.0, "🟢 Easy"),
            (8.0, "🟡 Moderate"),
            (10.0, "🟠 Hard"),
            (12.0, "🔴 Very Hard"),
            (float("inf"), "⛔ Nearly Impossible"),
        ],
        "ODI": [
            (4.5, "🟢 Easy"),
            (6.0, "🟡 Moderate"),
            (8.0, "🟠 Hard"),
            (10.0, "🔴 Very Hard"),
            (float("inf"), "⛔ Nearly Impossible"),
        ],
    }
    for threshold, label in thresholds.get(match_format, thresholds["T20"]):
        if required_rr <= threshold:
            return label
    return "⛔ Nearly Impossible"


def balls_to_overs(balls: int) -> str:
    """Convert ball count to 'X.Y' over notation."""
    return f"{balls // 6}.{balls % 6}"


def overs_to_balls(overs: float) -> int:
    """Convert decimal overs (e.g. 4.3) to total balls."""
    full = int(overs)
    partial = round((overs - full) * 10)
    return full * 6 + partial


def projected_score(current_runs: int, over: int, total_overs: int) -> int:
    """Simple linear projection of final score (no ML)."""
    if over <= 0:
        return 0
    crr = current_runs / over
    return int(crr * total_overs)


def win_probability_simple(
    target: int,
    current_runs: int,
    over: int,
    wickets: int,
    total_overs: int,
) -> float:
    """
    Heuristic win probability (used as a fallback/sanity check).
    Based on runs needed, required RR, and wickets in hand.
    """
    if current_runs >= target:
        return 100.0
    overs_rem = total_overs - over
    if overs_rem <= 0:
        return 0.0

    runs_needed = target - current_runs
    req_rr = runs_needed / overs_rem
    crr = current_runs / over if over > 0 else 0.0
    wickets_in_hand = 10 - wickets

    rr_factor = crr / req_rr if req_rr > 0 else 1.0
    wkt_factor = wickets_in_hand / 10.0

    raw = (rr_factor * 0.6 + wkt_factor * 0.4) * 50
    return round(max(2.0, min(98.0, raw)), 1)
