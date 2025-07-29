import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import itertools

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")
st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Load raw scorecard
df_raw = None
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_csv("mock_gto_scorecard_updated.csv")
    st.sidebar.info("Using default mock scorecard.")

# Sidebar builder settings
st.sidebar.header("Builder Settings")
min_ceiling = st.sidebar.slider("Min Ceiling (yards)", int(df_raw['Ceiling'].min()), int(df_raw['Ceiling'].max()), 65)
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", True)
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", True)
max_exposure_pct = st.sidebar.slider("Max Exposure (%)", 0.0, 100.0, 26.5, step=0.1)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range (49700-50000)", True)
# New double-punt controls
enforce_double = st.sidebar.checkbox("Enforce Max Double-Punts", True)
max_double = st.sidebar.slider("Max Lineups with Double-Punts", 0, 150, 15)
double_threshold = st.sidebar.slider("Double-Punt Ownership Threshold (%)", 0.0, 5.0, 2.75, step=0.1)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# Tabs layout
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Player Pool", "⚙️ Builder Settings", "🎲 Lineups", "📈 Ownership Report"
])

# Prepare player pool (NO FILTERING by ceiling or GTO%)
df_pool = df_raw.dropna(subset=["Name","Salary","GTO_Ownership%","Projected_Ownership%","Ceiling"]).reset_index(drop=True)
df_pool["Leverage"] = (df_pool["GTO_Ownership%"] / df_pool["Projected_Ownership%"]).round(1)
cols = list(df_pool.columns)
if "Leverage" in cols and "Salary" in cols:
    cols.remove("Leverage")
    idx = cols.index("Salary") + 1
    cols.insert(idx, "Leverage")
    df_pool = df_pool[cols]
with tab1:
    st.subheader("Player Pool (no filter)")
    st.dataframe(df_pool, use_container_width=True)

# Display settings
with tab2:
    st.subheader("Builder Settings")
    st.markdown(f"""
- **Min Ceiling:** {min_ceiling}
- **Singleton Rule:** {'✅' if enforce_singleton else '❌'}
- **GTO Weights:** {'✅' if enforce_weighting else '❌'}
- **Exposure Cap:** {'✅' if enforce_cap else '❌'} @ {max_exposure_pct}%
- **Salary Range:** {'✅' if enforce_salary else '❌'}
- **Double-Punt Cap:** {'✅' if enforce_double else '❌'} @ {max_double} lineups
- **Double-Punt Threshold:** {double_threshold}%
- **Lineups to Generate:** {total_lineups}
""" )

# Lineup builder logic
def build_lineups():
    # Setup
    pool = df_pool.copy()
    names = pool['Name'].tolist()
    pm = pool.set_index('Name').to_dict(orient='index')
    salary_range = (49700, 50000)
    max_exp = total_lineups * (max_exposure_pct/100)
    exposure = Counter()
    seen = set()
    lineups = []
    bias_count = 0
    double_count = 0

    def count_double(lu):
        return sum(1 for p in lu if pm[p]['GTO_Ownership%'] < double_threshold) >= 2

    def is_valid(lu):
        nonlocal double_count
        key = tuple(sorted(lu))
        if key in seen:
            return False
        if enforce_salary:
            s = sum(pm[n]['Salary'] for n in lu)
            if not (salary_range[0] <= s <= salary_range[1]):
                return False
        if enforce_cap and any(exposure[n] >= max_exp for n in lu):
            return False
        if enforce_double and double_count + (1 if count_double(lu) else 0) > max_double:
            return False
        return True

    def add(lu):
        nonlocal double_count
        seen.add(tuple(sorted(lu)))
        lineups.append(lu)
        if count_double(lu):
            double_count += 1
        for n in lu:
            exposure[n] += 1

    # Singleton enforcement
    if enforce_singleton:
        unused = set(names)
        while unused and len(lineups) < total_lineups:
            name = unused.pop()
            while True:
                others = [n for n in names if n != name]
                p = None
                if enforce_weighting:
                    w = [pm[n]['GTO_Ownership%'] for n in others]
                    tot = sum(w); p = [x/tot for x in w]
                cand = list(np.random.choice(others, 5, replace=False, p=p)) + [name]
                if is_valid(cand):
                    add(cand); break

    # Fill remaining lineups
    while len(lineups) < total_lineups:
        p = None
        if enforce_weighting:
            w = [pm[n]['GTO_Ownership%'] for n in names]
            tot = sum(w); p = [x/tot for x in w]
        cand = list(np.random.choice(names, 6, replace=False, p=p))
        if is_valid(cand):
            add(cand)

    return lineups, exposure

# Run simulation
def run():
    return build_lineups()

if 'lineups' not in st.session_state:
    st.session_state.lineups = None; st.session_state.exposure = None
if st.sidebar.button("Run Simulation"):
    with st.spinner("Generating lineups…"):
        lu, ex = run()
        st.session_state.lineups, st.session_state.exposure = lu, ex

# Lineups tab
with tab3:
    st.subheader("Lineups")
    if st.session_state.lineups:
        df_lu = pd.DataFrame([
            {f'PG{i+1}': p for i, p in enumerate(sorted(l))} for l in st.session_state.lineups
        ])
        st.dataframe(df_lu, use_container_width=True)
        st.download_button("📥 Download DraftKings CSV", df_lu.to_csv(index=False), file_name="gto_dk_upload.csv")
    else:
        st.info("Click 'Run Simulation' to generate lineups.")

# Ownership Report tab
with tab4:
    st.subheader("Ownership Exposure Summary")
    if st.session_state.exposure:
        exp_df = pd.DataFrame({
            'Name': list(st.session_state.exposure.keys()),
            'Lineup Count': list(st.session_state.exposure.values())
        })
        exp_df['Exposure %'] = exp_df['Lineup Count'] / total_lineups * 100
        st.dataframe(exp_df.sort_values('Exposure %', ascending=False), use_container_width=True)
    else:
        st.info("Click 'Run Simulation' to see exposure summary.")
