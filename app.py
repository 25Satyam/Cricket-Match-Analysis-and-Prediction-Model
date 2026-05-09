import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import CricketPredictor
from utils import (
    calculate_run_rate,
    calculate_required_run_rate,
    get_match_phase,
    get_chase_difficulty,
    format_score,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 Cricket Match Analysis & Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a472a;
        text-align: center;
        letter-spacing: 1px;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a472a;
        border-left: 4px solid #2d6a4f;
        padding-left: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .winner-box {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(match_format: str) -> tuple[CricketPredictor, float, float]:
    predictor = CricketPredictor()
    with st.spinner("⚙️  Training ML models on synthetic match data …"):
        mae, acc = predictor.train(match_format=match_format)
    return predictor, mae, acc


# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<div class="main-header">🏏 Cricket Match Analysis & Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">ML-powered score prediction & win probability — at any stage of the match</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️  Match Setup")

        match_format = st.selectbox("Match Format", ["T20", "ODI"])
        total_overs = 20 if match_format == "T20" else 50

        st.markdown("### 🔵 Team 1")
        team1 = st.text_input("Team 1 Name", "India")

        st.markdown("### 🔴 Team 2")
        team2 = st.text_input("Team 2 Name", "Australia")

        st.markdown("---")

        predictor, mae, acc = load_model(match_format)

        st.markdown("### 📊 Model Stats")
        st.metric("Score MAE", f"±{mae:.1f} runs")
        st.metric("Win Accuracy", f"{acc * 100:.1f} %")

        st.markdown("---")
        st.caption(
            "Built with Python · Streamlit · Pandas · NumPy · scikit-learn · Plotly"
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Score Predictor", "📈 Win Probability", "⚡ Live Simulator", "📊 Match Comparison"]
    )

    # ════════════════════════════
    # TAB 1 — Score Predictor
    # ════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Predict final innings score from current state</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            runs = st.number_input("Current Runs", 0, 500, 92, key="s_runs")
        with c2:
            wickets = st.number_input("Wickets Fallen", 0, 9, 3, key="s_wkts")
        with c3:
            over = st.number_input("Overs Completed", 1, total_overs - 1, 11, key="s_over")

        phase = get_match_phase(over, total_overs)
        crr = calculate_run_rate(runs, over)

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.info(f"**Match Phase:** {phase}")
        col_info2.info(f"**Current RR:** {crr}")
        col_info3.info(f"**Score:** {format_score(runs, wickets)}")

        if st.button("🔮 Predict Final Score", use_container_width=True, key="pred_score"):
            predicted, lower, upper = predictor.predict_score(over, runs, wickets, total_overs)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 Predicted", str(predicted))
            m2.metric("📉 Lower", str(lower))
            m3.metric("📈 Upper", str(upper))
            m4.metric("Runs to Bat", str(predicted - runs))

            # Gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted,
                    delta={"reference": runs, "increasing": {"color": "#2d6a4f"}},
                    title={"text": f"Predicted Final — {team1}", "font": {"size": 20}},
                    gauge={
                        "axis": {"range": [0, 300 if match_format == "T20" else 520]},
                        "bar": {"color": "#1a472a"},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [0, lower], "color": "#d8f3dc"},
                            {"range": [lower, upper], "color": "#95d5b2"},
                        ],
                        "threshold": {
                            "line": {"color": "#d62828", "width": 4},
                            "thickness": 0.8,
                            "value": predicted,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=320, margin=dict(t=60, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Remaining innings info
            rem_runs = predicted - runs
            rem_overs = total_overs - over
            rrr = calculate_required_run_rate(rem_runs, rem_overs)

            st.markdown("#### 📋 Remaining Innings Projection")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Runs Remaining", rem_runs)
            rc2.metric("Overs Remaining", rem_overs)
            rc3.metric("Projected RR", str(rrr))
            rc4.metric("Wickets in Hand", 10 - wickets)

    # ════════════════════════════
    # TAB 2 — Win Probability
    # ════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Win probability during a run chase</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            target = st.number_input("Target Score", 50, 500, 178, key="wp_target")
            chase_runs = st.number_input(f"{team2} Current Runs", 0, 500, 74, key="wp_runs")
        with c2:
            chase_wkts = st.number_input("Wickets Lost", 0, 9, 2, key="wp_wkts")
            chase_over = st.number_input("Overs Completed", 1, total_overs - 1, 9, key="wp_over")

        runs_needed = max(0, target - chase_runs)
        rem_overs = total_overs - chase_over
        rrr = calculate_required_run_rate(runs_needed, rem_overs)
        difficulty = get_chase_difficulty(rrr, match_format)

        di1, di2, di3 = st.columns(3)
        di1.info(f"**Runs Needed:** {runs_needed}")
        di2.info(f"**Required RR:** {rrr}")
        di3.info(f"**Chase Difficulty:** {difficulty}")

        if st.button("📊 Calculate Win Probability", use_container_width=True, key="calc_wp"):
            win_prob = predictor.predict_win_probability(
                target, chase_runs, chase_over, chase_wkts, total_overs
            )
            lose_prob = round(100 - win_prob, 1)

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric(f"🔴 {team2} Win %", f"{win_prob}%")
            wc2.metric(f"🔵 {team1} Win %", f"{lose_prob}%")
            wc3.metric("Overs Remaining", rem_overs)

            c_left, c_right = st.columns(2)

            # Donut
            with c_left:
                fig_donut = go.Figure(
                    go.Pie(
                        labels=[f"{team2} Wins", f"{team1} Wins"],
                        values=[win_prob, lose_prob],
                        hole=0.62,
                        marker_colors=["#e63946", "#457b9d"],
                        textinfo="label+percent",
                    )
                )
                fig_donut.update_layout(
                    title=f"Win Probability at Over {chase_over}",
                    height=340,
                    annotations=[
                        dict(
                            text=f"{win_prob}%",
                            x=0.5,
                            y=0.5,
                            font_size=26,
                            font_color="#e63946",
                            showarrow=False,
                        )
                    ],
                    showlegend=True,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # RR bar
            with c_right:
                crr_now = calculate_run_rate(chase_runs, chase_over)
                fig_rr = go.Figure(
                    go.Bar(
                        x=["Current RR", "Required RR"],
                        y=[crr_now, rrr],
                        marker_color=[
                            "#2d6a4f" if crr_now >= rrr else "#e63946",
                            "#457b9d",
                        ],
                        text=[f"{crr_now:.2f}", f"{rrr:.2f}"],
                        textposition="auto",
                        width=0.4,
                    )
                )
                fig_rr.update_layout(
                    title="Run Rate Comparison",
                    yaxis_title="Run Rate",
                    height=340,
                )
                st.plotly_chart(fig_rr, use_container_width=True)

    # ════════════════════════════
    # TAB 3 — Live Simulator
    # ════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Simulate over-by-over win probability graph for any target</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            sim_target = st.number_input("Target Score", 50, 500, 185, key="sim_tgt")
        with sc2:
            sim_seed = st.number_input("Random Seed (vary match scenarios)", 1, 999, 42, key="sim_seed")

        if st.button("▶️  Run Over-by-Over Simulation", use_container_width=True):
            with st.spinner("Simulating match …"):
                sim_df = predictor.simulate_over_by_over(sim_target, total_overs, seed=sim_seed)

            # Win probability graph
            fig_wp = go.Figure()
            fig_wp.add_trace(
                go.Scatter(
                    x=sim_df["over"],
                    y=sim_df["win_probability"],
                    mode="lines+markers",
                    name=f"{team2} (Chasing)",
                    line=dict(color="#e63946", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(230,57,70,0.1)",
                )
            )
            fig_wp.add_trace(
                go.Scatter(
                    x=sim_df["over"],
                    y=sim_df["lose_probability"],
                    mode="lines+markers",
                    name=f"{team1} (Defending)",
                    line=dict(color="#457b9d", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(69,123,157,0.1)",
                )
            )
            fig_wp.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="50% line")
            fig_wp.update_layout(
                title=f"Win Probability Over Time  (Target {sim_target})",
                xaxis_title="Over",
                yaxis_title="Win Probability (%)",
                yaxis=dict(range=[0, 100]),
                height=420,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_wp, use_container_width=True)

            # Runs progression
            fig_runs = go.Figure()
            fig_runs.add_trace(
                go.Scatter(
                    x=sim_df["over"],
                    y=sim_df["runs"],
                    mode="lines+markers",
                    name="Runs Scored",
                    line=dict(color="#2d6a4f", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(45,106,79,0.1)",
                )
            )
            fig_runs.add_hline(
                y=sim_target,
                line_dash="dot",
                line_color="#d62828",
                annotation_text=f"Target: {sim_target}",
            )
            fig_runs.update_layout(
                title="Runs Progression in Chase",
                xaxis_title="Over",
                yaxis_title="Cumulative Runs",
                height=300,
            )
            st.plotly_chart(fig_runs, use_container_width=True)

            # Wickets bar
            fig_wkts = go.Figure(
                go.Bar(
                    x=sim_df["over"],
                    y=sim_df["wickets"],
                    name="Cumulative Wickets",
                    marker_color="#e63946",
                )
            )
            fig_wkts.update_layout(
                title="Wickets Fallen", xaxis_title="Over", yaxis_title="Wickets", height=250
            )
            st.plotly_chart(fig_wkts, use_container_width=True)

            # Data table
            with st.expander("📋 Over-by-Over Data Table"):
                display_df = sim_df.copy()
                display_df.columns = ["Over", "Runs", "Wickets", "Win Prob (%)", "Defend Prob (%)"]
                st.dataframe(display_df, use_container_width=True)

    # ════════════════════════════
    # TAB 4 — Match Comparison
    # ════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Head-to-head innings comparison</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### 🔵 {team1}")
            t1_score = st.number_input("Score", 50, 500, 192, key="c_t1s")
            t1_wkts = st.number_input("Wickets Lost", 0, 10, 5, key="c_t1w")
            t1_sixes = st.number_input("Sixes", 0, 30, 9, key="c_t1six")
            t1_fours = st.number_input("Fours", 0, 50, 16, key="c_t1f")
            t1_dot = st.number_input("Dot Balls", 0, 120, 28, key="c_t1d")

        with c2:
            st.markdown(f"#### 🔴 {team2}")
            t2_score = st.number_input("Score", 50, 500, 171, key="c_t2s")
            t2_wkts = st.number_input("Wickets Lost", 0, 10, 7, key="c_t2w")
            t2_sixes = st.number_input("Sixes", 0, 30, 6, key="c_t2six")
            t2_fours = st.number_input("Fours", 0, 50, 12, key="c_t2f")
            t2_dot = st.number_input("Dot Balls", 0, 120, 35, key="c_t2d")

        if st.button("📊 Compare Teams", use_container_width=True):
            categories = ["Score", "Wickets Lost", "Sixes", "Fours", "Dot Balls"]
            t1_vals = [t1_score, t1_wkts, t1_sixes, t1_fours, t1_dot]
            t2_vals = [t2_score, t2_wkts, t2_sixes, t2_fours, t2_dot]

            # Grouped bar
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(name=team1, x=categories, y=t1_vals, marker_color="#457b9d")
            )
            fig_bar.add_trace(
                go.Bar(name=team2, x=categories, y=t2_vals, marker_color="#e63946")
            )
            fig_bar.update_layout(
                barmode="group",
                title="Head-to-Head Comparison",
                height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Radar
            max_v = [max(a, b, 1) for a, b in zip(t1_vals, t2_vals)]
            t1_n = [round(v / m * 100, 1) for v, m in zip(t1_vals, max_v)]
            t2_n = [round(v / m * 100, 1) for v, m in zip(t2_vals, max_v)]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=t1_n + [t1_n[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=team1,
                    line_color="#457b9d",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=t2_n + [t2_n[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=team2,
                    line_color="#e63946",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Performance Radar",
                height=400,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Winner banner
            margin = abs(t1_score - t2_score)
            if t1_score > t2_score:
                st.markdown(
                    f'<div class="winner-box">🏆 {team1} wins by {margin} runs!</div>',
                    unsafe_allow_html=True,
                )
            elif t2_score > t1_score:
                st.markdown(
                    f'<div class="winner-box">🏆 {team2} wins by {margin} runs!</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("🤝 Match Tied!")


if __name__ == "__main__":
    main()
