import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------------------------------
# SIMPLE EMBEDDED STYLE
# ----------------------------------------------------
st.markdown("""
<p style='text-align:center;
font-size:20px;
color:#374151;'>
Upload a CSV file to detect fraudulent transactions using Machine Learning.
</p>
""",
unsafe_allow_html=True
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload a CSV file to detect fraudulent transactions.")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    # Prepare data
    if "Class" in data.columns:
        y_true = data["Class"]
        X = data.drop(columns=["Class"])
    else:
        y_true = None
        X = data.copy()

    # Scale Amount
    if "Amount" in X.columns:
        X[["Amount"]] = scaler.transform(X[["Amount"]])

    # Drop Time
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])

    # Select Model
    model = rf_model if model_name == "Random Forest" else xgb_model

    # Predict
    prediction = model.predict(X)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:, 1]
    else:
        probability = [0] * len(prediction)

    # Result
    result = data.copy()
    result["Prediction"] = prediction
    result["Probability"] = probability

    result["Prediction"] = result["Prediction"].replace({
        0: "Genuine",
        1: "Fraud"
    })

    fraud = (prediction == 1).sum()
    genuine = (prediction == 0).sum()
    total = len(prediction)
    fraud_rate = round((fraud / total) * 100, 2)

    # ------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", total)
    c2.metric("Fraud", fraud)
    c3.metric("Genuine", genuine)
    c4.metric("Fraud %", f"{fraud_rate}%")
        # ==========================================================
    # CHARTS
    # ==========================================================

    st.markdown("---")
    st.subheader("📊 Prediction Analytics")

    col1, col2 = st.columns(2)

    # Donut Chart
    with col1:
        fig = px.pie(
            names=["Genuine", "Fraud"],
            values=[genuine, fraud],
            hole=0.6,
            color=["Genuine", "Fraud"],
            color_discrete_map={
                "Genuine": "green",
                "Fraud": "red"
            },
            title="Fraud vs Genuine"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    with col2:

        chart = pd.DataFrame({
            "Category": ["Genuine", "Fraud"],
            "Count": [genuine, fraud]
        })

        fig = px.bar(
            chart,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            color_discrete_map={
                "Genuine": "green",
                "Fraud": "red"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # HISTOGRAM
    # ==========================================================

    if "Amount" in result.columns:

        st.subheader("💰 Transaction Amount Distribution")

        fig = px.histogram(
            result,
            x="Amount",
            nbins=40,
            color_discrete_sequence=["royalblue"]
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    if model_name == "Random Forest":

        st.subheader("⭐ Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    if y_true is not None:

        st.subheader("🔥 Confusion Matrix")

        cm = confusion_matrix(y_true, prediction)

        fig, ax = plt.subplots(figsize=(5,5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Genuine","Fraud"]
        )

        disp.plot(ax=ax, cmap="Blues")

        st.pyplot(fig)

    # ==========================================================
    # ROC CURVE
    # ==========================================================

    if y_true is not None and hasattr(model, "predict_proba"):

        st.subheader("📈 ROC Curve")

        fpr, tpr, _ = roc_curve(
            y_true,
            probability
        )

        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC = {roc_auc:.3f}"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0,1],
                y=[0,1],
                mode="lines",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # RESULT TABLE
    # ==========================================================

    st.subheader("📋 Prediction Results")

    st.dataframe(
        result,
        use_container_width=True
    )

    # ==========================================================
    # DOWNLOAD
    # ==========================================================

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Results",
        csv,
        "prediction_results.csv",
        "text/csv"
    )

else:

    st.info("⬅ Upload a CSV file from the sidebar.")