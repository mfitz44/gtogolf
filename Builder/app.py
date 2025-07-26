import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")
st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Load raw scorecard
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv("mock_gto_scorecard_updated.csv")
    st.sidebar.info("Using default mock scorecard.")

# Sidebar builder settings
st.sidebar.header("Builder Settings")
min_ceiling = st.sidebar.slider("Min Ceiling (yards)", 0, 200, 65)
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", True)
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", True)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range (49700-50000)", True)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# Tabs for layout
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Player Pool", "⚙️ Builder Settings", "🎲 Lineups", "📈 Ownership Report"
])

# 1) Player Pool: live-filtered by slider
pool_df = raw_df.dropna(subset=["Name", "Salary", "GTO_Ownership%", "Projected_Ownership%", "Ceiling"])
pool_df = pool_df[pool_df["Ceiling"] >= min_ceiling]
pool_df = pool_df[pool_df["GTO_Ownership%"] > 0.5].reset_index(drop=True)
pool_df["Leverage"] = (pool_df["GTO_Ownership%"] / pool_df["Projected_Ownership%"]).round(1)
cols = list(pool_df.columns)
if "Leverage" in cols and "Salary" in cols:
    cols.remove("Leverage")
    idx = cols.index("Salary") + 1
    cols.insert(idx, "Leverage")
    pool_df = pool_df[cols]

with tab1:
    st.subheader(f"Player Pool (Ceiling ≥ {min_ceiling}, GTO > 0.5%)")
    st.dataframe(pool_df, use_container_width=True)

# 2) Builder Settings display
with tab2:
    st.subheader("Builder Settings")
    st.markdown(f"""
- **Min Ceiling:** {min_ceiling}
- **Singleton Rule:** {'✅' if enforce_singleton else '❌'}
- **GTO Weights:** {'✅' if enforce_weighting else '❌'}
- **Exposure Cap:** {'✅' if enforce_cap else '❌'}
- **Salary Range:** {'✅' if enforce_salary else '❌'}
- **Lineups to Generate:** {total_lineups}
""")

# 3) Cachable lineup generator
@st.cache_data(show_spinner=False)
def build_lineups(min_ceiling, enforce_singleton, enforce_weighting,
                  enforce_cap, enforce_salary, total_lineups):
    df = raw_df.copy()
    df = df.dropna(subset=["Name", "Salary", "GTO_Ownership%", "Projected_Ownership%", "Ceiling"])
    df = df[df["Ceiling"] >= min_ceiling]
    df = df[df["GTO_Ownership%"] > 0.5].reset_index(drop=True)
    names = df["Name"].tolist()
    player_map = df.set_index("Name").to_dict(orient="index")
    salary_range = (49700, 50000)
    max_exposure = total_lineups * 0.265
    exposure = Counter()
    seen = set()
    lineups = []
    unused = set(names)

    def is_valid(lu):
        key = tuple(sorted(lu))
        if key in seen:
            return False
        if enforce_salary:
            s = sum(player_map[n]["Salary"] for n in lu)
            if not (salary_range[0] <= s <= salary_range[1]):
                return False
        if enforce_cap:
            if any(exposure[n] >= max_exposure for n in lu):
                return False
        return True

    def add(lu):
        seen.add(tuple(sorted(lu)))
        for n in lu:
            exposure[n] += 1
            unused.discard(n)
        lineups.append(lu)

    # singleton rule
    if enforce_singleton:
        while unused:
            name = unused.pop()
            while True:
                others = [n for n in names if n != name]
                p = None
                if enforce_weighting:
                    w = [player_map[n]["GTO_Ownership%"] for n in others]
                    total = sum(w)
                    p = [x / total for x in w]
                chosen = list(np.random.choice(others, 5, replace=False, p=p))
                full = chosen + [name]
                if is_valid(full):
                    add(full)
                    break

    # fill remaining lineups
    while len(lineups) < total_lineups:
        p = None
        if enforce_weighting:
            w = [player_map[n]["GTO_Ownership%"] for n in names]
            total = sum(w)
            p = [x / total for x in w]
        chosen = list(np.random.choice(names, 6, replace=False, p=p))
        if is_valid(chosen):
            add(chosen)

    return lineups, exposure

# 4) Run Simulation button (only triggers on click)
if "lineups" not in st.session_state:
    st.session_state.lineups = None
    st.session_state.exposure = None

if st.sidebar.button("Run Simulation"):
    with st.spinner("⛳ Generating lineups…"):
        lu, ex = build_lineups(
            min_ceiling, enforce_singleton, enforce_weighting,
            enforce_cap, enforce_salary, total_lineups
        )
        st.session_state.lineups = lu
        st.session_state.exposure = ex

# 5) Lineups tab
with tab3:
    st.subheader("Lineups")
    if st.session_state.lineups:
        # display lineups
        lineup_df = pd.DataFrame([
            {"Lineup #": i+1, **{f"PG{j+1}": p for j, p in enumerate(sorted(lu))}}
            for i, lu in enumerate(st.session_state.lineups)
        ])
        st.dataframe(lineup_df, use_container_width=True)
        # download DraftKings CSV
        dk_df = pd.DataFrame([
            {f"PG{j+1}": p for j, p in enumerate(sorted(lu))}
            for lu in st.session_state.lineups
        ])
        st.download_button("📥 Download DraftKings CSV",
                           dk_df.to_csv(index=False),
                           file_name="gto_dk_upload.csv")
    else:
        st.info("Click 'Run Simulation' to generate lineups.")


# 6) Ownership Report tab
with tab4:
    st.subheader("Ownership Exposure Summary")
    if st.session_state.exposure:
        
        # Summary statistics
        total_pool = len(pool_df)
        used_players = exp_df['Name'].nunique()
        # Average lineup salary
        salary_map = raw_df.set_index('Name')['Salary'].to_dict()
        lineup_salaries = [sum(salary_map[n] for n in lu) for lu in st.session_state.lineups]
        avg_salary = sum(lineup_salaries) / len(lineup_salaries)
        st.markdown(f"**Players used:** {used_players} of {total_pool} in pool")
        st.markdown(f"**Average lineup salary:** ${avg_salary:,.0f}")
# Build summary DataFrame
        exp_df = pd.DataFrame({
            "Name": list(st.session_state.exposure.keys()),
            "Lineup Count": list(st.session_state.exposure.values()),
        })
        exp_df["Exposure %"] = exp_df["Lineup Count"] / total_lineups * 100

        # Merge in target and projected ownership from scorecard
        targets = raw_df.set_index("Name")[["GTO_Ownership%", "Projected_Ownership%"]]
        exp_df = exp_df.join(targets, on="Name")

        # Calculate targets and differences
        exp_df["Target Count"] = (exp_df["GTO_Ownership%"] / 100 * total_lineups).round(1)
        exp_df["Diff (Count)"] = exp_df["Lineup Count"] - exp_df["Target Count"]
        exp_df["Diff (%)"] = exp_df["Exposure %"] - exp_df["GTO_Ownership%"]

        # Reorder and format
        cols = [
            "Name", "Target Count", "Lineup Count", "Exposure %",
            "GTO_Ownership%", "Projected_Ownership%", "Diff (Count)", "Diff (%)"
        ]
        exp_df = exp_df[cols].sort_values("Exposure %", ascending=False).reset_index(drop=True)

        # Format percentages
        exp_df["Exposure %"] = exp_df["Exposure %"].map("{:.1f}%".format)
        exp_df["GTO_Ownership%"] = exp_df["GTO_Ownership%"].map("{:.1f}%".format)
        exp_df["Projected_Ownership%"] = exp_df["Projected_Ownership%"].map("{:.1f}%".format)
        exp_df["Diff (%)"] = exp_df["Diff (%)"].map("{:+.1f}%".format)

        st.dataframe(exp_df, use_container_width=True)

        # Download button
        st.download_button(
            "📥 Download Ownership Summary",
            exp_df.to_csv(index=False),
            file_name="ownership_summary.csv"
        )
    else:
        st.info("Click 'Run Simulation' to see exposure summary.")

with tab4:
    st.subheader("Ownership Exposure Summary")
    if st.session_state.exposure:
        
        # Summary statistics
        total_pool = len(pool_df)
        used_players = exp_df['Name'].nunique()
        # Average lineup salary
        salary_map = raw_df.set_index('Name')['Salary'].to_dict()
        lineup_salaries = [sum(salary_map[n] for n in lu) for lu in st.session_state.lineups]
        avg_salary = sum(lineup_salaries) / len(lineup_salaries)
        st.markdown(f"**Players used:** {used_players} of {total_pool} in pool")
        st.markdown(f"**Average lineup salary:** ${avg_salary:,.0f}")
exp_df = pd.DataFrame({
            "Name": list(st.session_state.exposure.keys()),
            "Lineup Count": list(st.session_state.exposure.values()),
            "Exposure %": [v / total_lineups * 100 for v in st.session_state.exposure.values()]
        }).sort_values("Exposure %", ascending=False)
        st.dataframe(exp_df.style.format({"Exposure %": "{:.1f}%"}),
                     use_container_width=True)
    else:
        st.info("Click 'Run Simulation' to see exposure summary.")
