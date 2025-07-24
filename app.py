import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")

st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Upload or use default scorecard
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mock_gto_scorecard.csv")
    st.sidebar.info("Using default mock scorecard.")

# Drop invalid rows
df = df.dropna(subset=["Name", "Salary", "GTO_Ownership%"])
df = df[df["GTO_Ownership%"] > 0.5].reset_index(drop=True)

# UI options
st.sidebar.header("Builder Settings")
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", value=True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", value=True)
enforce_cap = st.sidebar.checkbox("Enforce Max 26.5% Exposure", value=True)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range ($49,700–$50,000)", value=True)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# Setup
names = df["Name"].tolist()
weights = df["GTO_Ownership%"].values / df["GTO_Ownership%"].sum()
player_map = {row["Name"]: row for _, row in df.iterrows()}
salary_range = (49700, 50000)
max_exposure = 0.265
max_per_player = int(total_lineups * max_exposure)

def build_lineups():
    exposure = Counter()
    seen = set()
    lineups = []

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

    if enforce_singleton:
        for name in names:
            tries = 0
            while tries < 1000:
                others = [n for n in names if n != name]
                wts = [player_map[n]["GTO_Ownership%"] for n in others]
                total = sum(wts)
                wts = [w / total for w in wts] if enforce_weighting else None
                chosen = list(np.random.choice(others, 5, replace=False, p=wts))
                full = chosen + [name]
                if len(set(full)) != 6:
                    tries += 1
                    continue
                if is_valid(full):
                    add(full)
                    break
                tries += 1

    while len(lineups) < total_lineups:
        tries = 0
        while tries < 1000:
            chosen = list(np.random.choice(
                names, 6, replace=False, p=weights if enforce_weighting else None
            ))
            if is_valid(chosen):
                add(chosen)
                break
            tries += 1
        if tries >= 1000:
            break

    return lineups, exposure

# Build
st.header("🔄 Generating Lineups...")
final_lineups, exposure_counter = build_lineups()

# Lineup table
lineup_table = []
for idx, lineup in enumerate(final_lineups):
    total_salary = sum(player_map[n]["Salary"] for n in lineup)
    total_proj = sum(player_map[n]["ProjectedPoints"] for n in lineup)
    lineup_table.append({
        "Lineup #": idx + 1,
        "Players": ", ".join(sorted(lineup)),
        "Salary": total_salary,
        "Projected Points": total_proj
    })
lineup_df = pd.DataFrame(lineup_table)

# Exposure breakdown
exposure_df = pd.DataFrame({
    "Name": list(exposure_counter.keys()),
    "Lineup Count": list(exposure_counter.values()),
    "Exposure %": [v / total_lineups * 100 for v in exposure_counter.values()]
}).sort_values(by="Exposure %", ascending=False)

# Display
st.subheader("📊 Lineup Table")
st.dataframe(lineup_df)

st.subheader("🧮 Ownership Exposure Summary")
st.dataframe(exposure_df)

# Export
st.download_button("📥 Download Lineups CSV", lineup_df.to_csv(index=False), file_name="gto_lineups.csv")
