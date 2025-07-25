import streamlit as st
import pandas as pd
import numpy as np
import glob
from collections import Counter
import itertools

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")
st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Load or upload GTO scorecard
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv("mock_gto_scorecard_updated.csv")
    st.sidebar.info("Using default mock scorecard.")

# Determine dynamic ceiling range
max_ceiling_val = int(raw_df['Ceiling'].max())
min_ceiling = st.sidebar.slider(
    "Min Ceiling", 
    min_value=int(raw_df['Ceiling'].min()), 
    max_value=max_ceiling_val, 
    value=65,
    step=1,
    help=f"Select minimum ceiling between {int(raw_df['Ceiling'].min())} and {max_ceiling_val}"
)

# Sidebar builder settings
st.sidebar.header("Builder Settings")
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", True)
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", True)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range (49700-50000)", True)
max_bias_points = st.sidebar.slider("Max High-Bias Pair Occurrences per Build", 0, 100, 10)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# Automatic high-bias calculation
@st.cache_data
def calculate_high_bias_pairs():
    files = glob.glob("historical_builds/*.csv")
    if not files:
        return set()
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    player_cols = [c for c in combined.columns if c.lower() not in ['lineupname','entry']]
    total_lineups = combined.shape[0]
    counts = combined[player_cols].apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    high_bias = set()
    for p1, p2 in itertools.combinations(counts.index, 2):
        exp = (counts[p1]/total_lineups)*(counts[p2]/total_lineups)*total_lineups
        p = (counts[p1]/total_lineups)*(counts[p2]/total_lineups)
        std = np.sqrt(total_lineups * p * (1 - p))
        actual = combined[player_cols].apply(lambda r: p1 in r.values and p2 in r.values, axis=1).sum()
        if std > 0 and (actual - exp)/std > 3:
            high_bias.add(frozenset((p1, p2)))
    return high_bias

high_bias_pairs = calculate_high_bias_pairs()

# Clean & filter player pool
df = raw_df.dropna(subset=["Name","Salary","GTO_Ownership%","Projected_Ownership%","Ceiling"])
df = df[df["Ceiling"] >= min_ceiling]
df = df[df["GTO_Ownership%"] > 0.5].reset_index(drop=True)
if df.empty:
    st.warning("No players meet the selected ceiling threshold. Please lower the Min Ceiling.")

# Compute Leverage metric
df["Leverage"] = (df["GTO_Ownership%"] / df["Projected_Ownership%"]).round(1)

# Prepare builder inputs
names = df["Name"].tolist()
weights = df["GTO_Ownership%"].values / df["GTO_Ownership%"].sum()
player_map = {r["Name"]: r for _, r in df.iterrows()}
salary_range = (49700, 50000)
max_exposure = 0.265
max_per_player = int(total_lineups * max_exposure)

@st.cache_data
def build_lineups():
    exposure = Counter()
    seen = set()
    lineups = []
    unused = set(names)
    bias_count = 0

    def count_bias(lu):
        return sum(1 for pair in itertools.combinations(lu, 2) if frozenset(pair) in high_bias_pairs)

    def valid(lu):
        nonlocal bias_count
        key = tuple(sorted(lu))
        if key in seen: return False
        if enforce_salary and not (salary_range[0] <= sum(player_map[n]["Salary"] for n in lu) <= salary_range[1]): return False
        if enforce_cap and any(exposure[n] >= max_per_player for n in lu): return False
        if bias_count + count_bias(lu) > max_bias_points: return False
        return True

    def add(lu):
        nonlocal bias_count
        key = tuple(sorted(lu))
        seen.add(key)
        lineups.append(lu)
        for n in lu:
            exposure[n] += 1
            unused.discard(n)
        bias_count += count_bias(lu)

    # Singleton enforcement
    while enforce_singleton and unused:
        name = unused.pop()
        success = False
        while not success:
            choices = [n for n in names if n != name]
            probs = [player_map[n]["GTO_Ownership%"] for n in choices]
            total = sum(probs)
            p = [v/total for v in probs] if enforce_weighting else None
            cand = list(np.random.choice(choices, 5, replace=False, p=p)) + [name]
            if valid(cand): add(cand); success = True

    # Fill to total lineups
    while len(lineups) < total_lineups:
        cand = list(np.random.choice(names, 6, replace=False, p=weights if enforce_weighting else None))
        if valid(cand): add(cand)

    return lineups, exposure

if st.sidebar.button("Run Builder"):
    final_lineups, exposure_counter = build_lineups()
    st.write(f"Generated {len(final_lineups)} lineups.")
    # Additional display logic here...
