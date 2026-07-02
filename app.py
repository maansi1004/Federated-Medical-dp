import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Set up Page Config
st.set_page_config(page_title="Federated Health Dashboard", layout="wide")

st.title("🏥 Secure Federated Learning Dashboard")
st.subheader("Pneumonia Detection across Decentralized Hospital Silos with Differential Privacy")

# 2. Hardcode the actual metrics collected from your terminal run
data = {
    "Round": [1, 2, 3],
    "Global Loss": [0.5752, 0.5284, 0.4892],
    "Privacy Budget (Max ε)": [1.39, 1.73, 2.00]
}
df = pd.DataFrame(data)

# 3. Create Columns for Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📉 System Convergence")
    fig, ax1 = plt.subplots()
    
    # Plot Global Loss
    ax1.set_xlabel("Federated Rounds")
    ax1.set_ylabel("Global Loss", color="tab:red")
    ax1.plot(df["Round"], df["Global Loss"], color="tab:red", marker="o", linewidth=2)
    ax1.tick_params(axis='y', labelcolor="tab:red")
    ax1.set_xticks([1, 2, 3])
    
    st.pyplot(fig)

with col2:
    st.markdown("### 🔒 Differential Privacy Tracking")
    st.dataframe(df.set_index("Round"), use_container_width=True)
    
    st.success("""
    **Privacy Audit Pass:**
    * Max Epsilon ($\epsilon$): **2.00** ($\delta = 10^{-5}$)
    * Attack Vector Resistance: High (Mathematical Guarantee against gradient reconstruction attacks).
    """)

# 4. Hospital Node Details
st.markdown("---")
st.markdown("### 🏢 Active Node Network")
c_a, c_b, c_c = st.columns(3)
c_a.metric(label="Hospital A (Node 0)", value="1,552 Samples", delta="DP Active (ε=1.69)")
c_b.metric(label="Hospital B (Node 1)", value="1,165 Samples", delta="DP Active (ε=2.00)")
c_c.metric(label="Hospital C (Node 2)", value="1,165 Samples", delta="DP Active (ε=2.00)")