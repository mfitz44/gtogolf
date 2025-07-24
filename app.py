import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")
st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Upload or load default scorecard
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mock_gto_scorecard.csv")
    st.sidebar.info("Using default mock scorecard.")

# Clean & filter player pool
df = df.dropna(subset=["Name", "Salary", "GTO_Ownership%"])
df = df[df["GTO_Ownership%"] > 0.5].reset_index(drop=True)

# Sidebar builder toggles
st.sidebar.header("Builder Settings")
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", value=True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", value=True)
enforce_cap = st.sidebar.checkbox("Enforce Max 26.5% Exposure", value=True)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range ($49,700–$50,000)", value=True)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)
generate = st.sidebar.button("Generate Lineups")

# Setup
names = df["Name"].tolist()
weights = df["GTO_Ownership%"].values / df["GTO_Ownership%"].sum()
player_map = {row["Name"]: row for _, row in df.iterrows()}
salary_range = (49700, 50000)
max_exposure = 0.265
max_per_player = int(total_lineups * max_exposure)

# Lineup builder
def build_lineups():
    exposure = Counter()
    seen = set()
    lineups = []
    unused_players = set(names)

    def is_valid(lineup):
        key = tuple(sorted(lineup))
        if key in seen:
            return False
        if enforce_salary:
            s = sum(player_map[n]["Salary"] for n in lineup)
            if not (salary_range[0] <= s <= salary_range[1]):
                return False
        if enforce_cap:
            if any(exposure[n] >= max_per_player for n in lineup):
                return False
        return True

    def add(lineup):
        key = tuple(sorted(lineup))
        seen.add(key)
        lineups.append(lineup)
        for n in lineup:
            exposure[n] += 1
            unused_players.discard(n)

    # Strict Singleton Enforcement
    while unused_players:
        name = unused_players.pop()
        success = False
        while not success:
            others = [n for n in names if n != name]
            wts = [player_map[n]["GTO_Ownership%"] for n in others]
            total = sum(wts)
            wts = [w / total for w in wts] if enforce_weighting else None
            chosen = list(np.random.choice(others, 5, replace=False, p=wts))
            full = chosen + [name]
            if len(set(full)) == 6 and is_valid(full):
                add(full)
                success = True

    # Fill to full lineup count
    while len(lineups) < total_lineups:
        chosen = list(np.random.choice(
            names, 6, replace=False, p=weights if enforce_weighting else None
        ))
        if is_valid(chosen):
            add(chosen)

    return lineups, exposure

# Run builder
if generate:
        final_lineups, exposure_counter = build_lineups()

    # Format lineups
        lineup_table = []
        dk_export = []
        for idx, lineup in enumerate(final_lineups):
        total_salary = sum(player_map[n]["Salary"] for n in lineup)
        total_proj = sum(player_map[n]["ProjectedPoints"] for n in lineup)
        lineup_table.append({
            "Lineup #": idx + 1,
            "Players": ", ".join(sorted(lineup)),
            "Salary": total_salary,
            "Projected Points": total_proj
        })
        dk_export.append({f"PG{i+1}": p for i, p in enumerate(sorted(lineup))})

        lineup_df = pd.DataFrame(lineup_table)
        dk_df = pd.DataFrame(dk_export)

    # Exposure table
        exposure_df = pd.DataFrame({
        "Name": list(exposure_counter.keys()),
        "Lineup Count": list(exposure_counter.values()),
        "Exposure %": [v / total_lineups * 100 for v in exposure_counter.values()]
        }).sort_values(by="Exposure %", ascending=False)

    # Summary stats
        num_golfers_in_pool = len(df)
        num_golfers_used = len(set(name for lineup in final_lineups for name in lineup))
        avg_salary = lineup_df["Salary"].mean()
        min_proj = lineup_df["Projected Points"].min()
        max_proj = lineup_df["Projected Points"].max()

    # Build tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📥 Player Pool", "⚙️ Builder Settings", "📊 Lineups", "📈 Ownership Report"])

        with tab1:
        st.subheader("Player Pool (Filtered > 0.5% GTO Ownership)")
        st.dataframe(df, use_container_width=True)

        with tab2:
        st.subheader("Current Build Settings")
        st.markdown(f"""
        - Singleton Rule: {'✅ Enabled' if enforce_singleton else '❌ Off'}  
        - GTO Weighting: {'✅ Enabled' if enforce_weighting else '❌ Off'}  
        - Exposure Cap (26.5%): {'✅ Enabled' if enforce_cap else '❌ Off'}  
        - Salary Range ($49,700–$50,000): {'✅ Enabled' if enforce_salary else '❌ Off'}  
        - Total Lineups: `{total_lineups}`
        """)

        with tab3:
        st.subheader("Generated Lineups")
        st.dataframe(lineup_df.style.format({
            "Salary": "${:,.0f}",
            "Projected Points": "{:.1f}"
        }), use_container_width=True)
        st.download_button("📥 Download DraftKings CSV", dk_df.to_csv(index=False), file_name="gto_dk_upload.csv")

        with tab4:
        st.subheader("Ownership Exposure Summary")
        st.markdown(f"""
        - **Golfers in Pool:** {num_golfers_in_pool}  
        - **Golfers Used in Lineups:** {num_golfers_used}  
        - **Average Lineup Salary:** ${avg_salary:,.0f}  
        - **Projected Points Range:** {min_proj:.1f} – {max_proj:.1f}
        """)
        st.dataframe(exposure_df.style.format({
            "Exposure %": "{:.1f}%"
        }), use_container_width=True)
