import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# =========================
# PAGE CONFIG & STYLING
# =========================
st.set_page_config(page_title="IoT Shield | Botnet Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Botnet Detection Dashboard")
st.markdown("Real-time behavioral analysis of IoT network traffic (Danmini Doorbell).")
st.markdown("---")

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2885/2885412.png", width=100) # Optional placeholder icon
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload Network Traffic (CSV)", type=["csv"])
    st.markdown("---")
    st.info("Upload a CSV file containing extracted network features to run the anomaly detection model.")

# =========================
# MAIN DASHBOARD LOGIC
# =========================
# Load Model (Add caching so it doesn't reload on every UI click)
@st.cache_resource
def load_model():
    # Replace with your actual model path
    # return joblib.load("outputs/models/iot_model.pkl") 
    pass 

model = load_model()

if uploaded_file is not None:
    # 1. Read Data
    df = pd.read_csv(uploaded_file)
    original_df = df.copy() # Keep a copy for display
    
    # Check if this is test data (has labels) or new data
    has_labels = "label" in df.columns
    if has_labels:
        true_labels = df["label"]
        df = df.drop(columns=["label"])
    
    try:
        # 2. Make Predictions
        # predictions = model.predict(df)
        # probabilities = model.predict_proba(df)[:, 1]
        
        # --- MOCK DATA FOR PREVIEW PURPOSES (Remove these 2 lines in production) ---
        import numpy as np
        predictions = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3]) 
        probabilities = np.random.rand(len(df))
        # --------------------------------------------------------------------------

        df["Risk_Score"] = probabilities
        df["Prediction"] = predictions
        df["Status"] = df["Prediction"].map({0: "Benign", 1: "Attack"})

        # 3. Calculate Metrics
        total = len(df)
        attack_count = int((df["Prediction"] == 1).sum())
        benign_count = int((df["Prediction"] == 0).sum())
        attack_ratio = (attack_count / total) * 100

        # =========================
        # TOP ROW: METRICS & ALERT
        # =========================
        if attack_ratio > 50:
            st.error(f"🚨 CRITICAL ALERT: High volume of malicious traffic detected! ({attack_ratio:.1f}% of network flows)")
        elif attack_ratio > 10:
            st.warning(f"⚠️ WARNING: Suspicious activity detected. ({attack_ratio:.1f}% of network flows)")
        else:
            st.success(f"✅ SECURE: Network traffic appears normal. ({100 - attack_ratio:.1f}% benign flows)")

        st.write("") # Spacer

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Packets Analyzed", f"{total:,}")
        col2.metric("Benign Traffic", f"{benign_count:,}", delta="Safe", delta_color="normal")
        col3.metric("Attack Traffic", f"{attack_count:,}", delta="Threat", delta_color="inverse")
        col4.metric("Threat Level", f"{attack_ratio:.1f}%")

        st.markdown("---")

        # =========================
        # MIDDLE ROW: CHARTS (PLOTLY)
        # =========================
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Traffic Distribution")
            # Sleek Donut Chart instead of a blocky bar chart
            fig_pie = px.pie(
                names=["Benign", "Attack"], 
                values=[benign_count, attack_count],
                hole=0.6,
                color=["Benign", "Attack"],
                color_discrete_sequence=["#00CC96", "#EF553B"] # Green and Red
            )
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.subheader("Threat Gauge")
            # Professional Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=attack_ratio,
                title={'text': "Network Risk %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "gold"},
                        {'range': [50, 100], 'color': "salmon"}
                    ]
                }
            ))
            fig_gauge.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # =========================
        # BOTTOM ROW: DATA & ROC
        # =========================
        data_col, roc_col = st.columns([1.5, 1])

        with data_col:
            st.subheader("📄 Network Flow Log")
            # Show the data with the new prediction columns
            display_df = original_df.copy()
            display_df["Classification"] = df["Status"]
            display_df["Risk Score"] = df["Risk_Score"].round(3)
            
            # Style the dataframe so Attack rows are highlighted
            def color_threats(val):
                color = '#ffb3b3' if val == 'Attack' else ''
                return f'background-color: {color}'
            
            st.dataframe(display_df.head(100).style.map(color_threats, subset=['Classification']), use_container_width=True, height=350)

        with roc_col:
            st.subheader("Model Performance (ROC)")
            if has_labels:
                # Calculate ROC only if we have the true labels
                fpr, tpr, _ = roc_curve(true_labels, probabilities)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (AUC={roc_auc:.3f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate')
                )
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=350)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ℹ️ Upload a dataset containing a 'label' column (ground truth) to view the ROC curve.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    # What to show when no file is uploaded yet
    st.info("👈 Please upload a CSV file from the sidebar to begin analysis.")
    st.image("https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=2000&auto=format&fit=crop", caption="Awaiting Network Telemetry...", use_column_width=True)