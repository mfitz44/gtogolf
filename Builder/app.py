import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="GTO PGA DFS Lineup Builder", layout="wide")

st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Upload scorecard or use default
st.sidebar.header("Upload GTO Scorecard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mock_gto_scorecard.csv")
    st.sidebar.info("Using default mock data.")

# Preview scorecard
st.subheader("📋 Player Pool (From GTO Scorecard)")
st.dataframe(df)

# Filter out players with 0.5% or less GTO ownership
df = df[df["GTO_Ownership%"] > 0.5].reset_index(drop=True)

# User configuration
st.sidebar.header("Lineup Builder Settings")
lineup_count = st.sidebar.slider("Number of Lineups", 1, 150, 150)
min_salary = st.sidebar.number_input("Min Salary", value=49700)
max_salary = st.sidebar.number_input("Max Salary", value=50000)
max_exposure = st.sidebar.slider("Max Exposure per Player (%)", 5.0, 50.0, 26.5)
ownership_tolerance = st.sidebar.slider("Ownership Tolerance ±%", 0.0, 5.0, 2.0)

# Build lineups
st.header("🛠️ Generated Lineups")
lineups = []
used_once = set()
player_pool = df.to_dict("records")

while len(lineups) < lineup_count:
    lineup = random.sample(player_pool, 6)
    salary = sum(p["Salary"] for p in lineup)
    if not (min_salary <= salary <= max_salary):
        continue
    lineup_names = tuple(sorted(p["Name"] for p in lineup))
    if lineup_names in lineups:
        continue
    lineups.append(lineup_names)
    for p in lineup:
        used_once.add(p["Name"])
    if len(used_once) == len(player_pool):
        break

# Show lineups
lineup_table = []
for idx, lineup in enumerate(lineups):
    total_salary = sum(df[df["Name"] == name]["Salary"].values[0] for name in lineup)
    total_proj = sum(df[df["Name"] == name]["ProjectedPoints"].values[0] for name in lineup)
    lineup_table.append({"Lineup #": idx + 1, "Players": ", ".join(lineup), "Salary": total_salary, "ProjPts": total_proj})

st.dataframe(pd.DataFrame(lineup_table))

# Export
st.download_button("📥 Download Lineups CSV", pd.DataFrame(lineup_table).to_csv(index=False), file_name="gto_lineups.csv")
