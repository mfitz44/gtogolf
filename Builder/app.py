import streamlit as st
import pandas as pd
import numpy as np
import glob
from collections import Counter
import itertools

st.set_page_config(page_title="GTO PGA Lineup Builder", layout="wide")
st.title("🏌️‍♂️ GTO PGA DFS Lineup Builder")

# Load raw scorecard
global_pool = None
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv("mock_gto_scorecard_updated.csv")
    st.sidebar.info("Using default mock scorecard.")

# Sidebar builder settings
st.sidebar.header("Builder Settings")
min_ceiling = st.sidebar.slider("Min Ceiling (yards)", int(raw_df['Ceiling'].min()), int(raw_df['Ceiling'].max()), 65)
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule", True)
enforce_weighting = st.sidebar.checkbox("Use GTO Ownership Weights", True)
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", True)
max_exposure_pct = st.sidebar.slider("Max Exposure (%)", 0.0, 100.0, 26.5, step=0.1)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range (49700-50000)", True)
enforce_bias = st.sidebar.checkbox("Enforce Bias-Points Cap", True)
max_bias_points = st.sidebar.slider("Max Bias Points per Build", 0, 50, 10)
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# 1) Player Pool
tab1, tab2, tab3, tab4 = st.tabs(["👥 Player Pool", "⚙️ Builder Settings", "🎲 Lineups", "📈 Ownership Report"])
pool = raw_df.dropna(subset=["Name","Salary","GTO_Ownership%","Projected_Ownership%","Ceiling"])
pool = pool[pool["Ceiling"] >= min_ceiling]
pool = pool[pool["GTO_Ownership%"] > 0.5].reset_index(drop=True)
pool["Leverage"] = (pool["GTO_Ownership%"] / pool["Projected_Ownership%"]).round(1)
cols = list(pool.columns)
if "Leverage" in cols and "Salary" in cols:
    cols.remove("Leverage")
    idx = cols.index("Salary") + 1
    cols.insert(idx, "Leverage")
    pool = pool[cols]
with tab1:
    st.subheader(f"Player Pool (Ceiling ≥ {min_ceiling}, GTO > 0.5%)")
    st.dataframe(pool, use_container_width=True)

# 2) Builder Settings display
with tab2:
    st.subheader("Builder Settings")
    settings_md = f"""
- **Min Ceiling:** {min_ceiling}  
- **Singleton Rule:** {'✅' if enforce_singleton else '❌'}  
- **GTO Weights:** {'✅' if enforce_weighting else '❌'}  
- **Exposure Cap:** {'✅' if enforce_cap else '❌'} @ {max_exposure_pct}%  
- **Salary Range:** {'✅' if enforce_salary else '❌'}  
- **Bias-Points Cap:** {'✅' if enforce_bias else '❌'} @ {max_bias_points} pts  
- **Lineups:** {total_lineups}
"""
    st.markdown(settings_md)

# Compute high-bias pairs via Monte Carlo sampling of pool
@st.cache_data
def get_high_bias_pairs_from_pool(pool_df, trials=10000):
    names = pool_df['Name'].tolist()
    samples = [tuple(sorted(np.random.choice(names, 6, replace=False))) for _ in range(trials)]
    df_samples = pd.DataFrame(samples, columns=[f'PG{i+1}' for i in range(6)])
    counts = df_samples.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    n = trials
    bias_pairs = set()
    for p1, p2 in itertools.combinations(counts.index, 2):
        exp = (counts[p1]/n) * (counts[p2]/n) * n
        p_prob = (counts[p1]/n) * (counts[p2]/n)
        std = np.sqrt(n * p_prob * (1 - p_prob))
        actual = df_samples.apply(lambda r: p1 in r.values and p2 in r.values, axis=1).sum()
        if std > 0 and (actual - exp)/std > 3:
            bias_pairs.add(frozenset((p1, p2)))
    return bias_pairs

high_bias_pairs = get_high_bias_pairs_from_pool(pool)

# Builder function
def build_lineups():
    pm = pool.set_index('Name').to_dict(orient='index')
    salary_range = (49700, 50000)
    max_exp = total_lineups * (max_exposure_pct/100)
    exposure = Counter(); seen = set(); lineups = []; bias_count = 0
    unused = set(pool['Name'].tolist())

    def count_bias(lu):
        return sum(1 for c in itertools.combinations(lu, 2) if frozenset(c) in high_bias_pairs)
    def valid(lu):
        nonlocal bias_count
        key = tuple(sorted(lu))
        if key in seen: return False
        if enforce_salary:
            s = sum(pm[n]['Salary'] for n in lu)
            if not (salary_range[0] <= s <= salary_range[1]): return False
        if enforce_cap and any(exposure[n] >= max_exp for n in lu): return False
        if enforce_bias and (bias_count + count_bias(lu) > max_bias_points): return False
        return True
    def add(lu):
        nonlocal bias_count
        seen.add(tuple(sorted(lu))); lineups.append(lu)
        for n in lu:
            exposure[n] += 1; unused.discard(n)
        bias_count += count_bias(lu)

    # Singleton enforcement
    if enforce_singleton:
        while unused:
            name = unused.pop()
            while True:
                others = [n for n in pool['Name'] if n != name]
                if enforce_weighting:
                    w = [pm[n]['GTO_Ownership%'] for n in others]
                    tot = sum(w); probs = [x/tot for x in w]
                else:
                    probs = None
                cand = list(np.random.choice(others, 5, replace=False, p=probs)) + [name]
                if valid(cand): add(cand); break
    # Fill remaining
    while len(lineups) < total_lineups:
        if enforce_weighting:
            w = [pm[n]['GTO_Ownership%'] for n in pool['Name']]
            tot = sum(w); probs = [x/tot for x in w]
            cand = list(np.random.choice(pool['Name'], 6, replace=False, p=probs))
        else:
            cand = list(np.random.choice(pool['Name'], 6, replace=False))
        if valid(cand): add(cand)
    return lineups, exposure

# Simulation trigger
if 'lineups' not in st.session_state:
    st.session_state.lineups = None; st.session_state.exposure = None
if st.sidebar.button("Run Simulation"):
    with st.spinner("Generating lineups…"):
        lu, ex = build_lineups()
        st.session_state.lineups, st.session_state.exposure = lu, ex

# 3) Lineups Tab
with tab3:
    st.subheader("Generated Lineups")
    if st.session_state.lineups:
        df_lu = pd.DataFrame([{f'PG{i+1}': p for i, p in enumerate(sorted(l))} for l in st.session_state.lineups])
        st.dataframe(df_lu, use_container_width=True)
        st.download_button("📥 Download DraftKings CSV", df_lu.to_csv(index=False), file_name="gto_dk_upload.csv")
    else:
        st.info("Click 'Run Simulation' to generate lineups.")

# 4) Ownership Report Tab
with tab4:
    st.subheader("Ownership Exposure Summary")
    if st.session_state.exposure:
        used = st.session_state.exposure
        st.markdown(f"**Players used:** {len(used)} / {len(pool)}")
        exp_df = pd.DataFrame({'Name': list(used.keys()), 'Count': list(used.values())})
        exp_df['Exposure %'] = exp_df['Count'] / total_lineups * 100
        st.dataframe(exp_df.sort_values('Exposure %', ascending=False), use_container_width=True)
    else:
        st.info("Click 'Run Simulation' to see exposure summary.")
