import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# --- Page Configuration ---
st.set_page_config(
    page_title="Exoplanet Analysis Dashboard",
    page_icon="🛰️",
    layout="wide"
)

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    """Load the trained pipeline, feature columns, full dataset, and metrics."""
    try:
        pipeline = joblib.load('exoplanet_ensemble_model.joblib')
        columns = joblib.load('feature_columns.joblib')
        metrics = joblib.load('training_metrics.joblib')
        df_full = pd.read_csv('exoplanet_data_merged_for_ensemble.csv')
        return pipeline, columns, df_full, metrics
    except FileNotFoundError:
        st.error("Model assets not found. Please run `data_preparation.py` and `train_model.py` first.")
        return None, None, None, None

pipeline, feature_columns, df_full, metrics = load_assets()
disposition_map_inv = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
class_names = list(disposition_map_inv.values())

# --- Helper Functions for Plotting ---
def plot_confusion_matrix(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        metrics['y_test'],
        metrics['y_pred'],
        display_labels=class_names,
        cmap='Blues',
        ax=ax
    )
    ax.set_title("Confusion Matrix on Test Set")
    st.pyplot(fig)

def plot_feature_importance(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='importance',
        y='feature',
        data=metrics['feature_importances'],
        palette='viridis',
        ax=ax
    )
    ax.set_title("Average Feature Importance Across Ensemble")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


# --- UI Layout ---
st.title("🛰️ Exoplanet Analysis & Classification Dashboard")
st.write("""
This tool uses a high-performance **Ensemble Model** trained on a merged dataset from the **Kepler, K2, and TESS missions**. 
Use the tabs below to perform live classifications or to explore the model's performance metrics from its last training run.
""")

if pipeline:
    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Live Classification", "📈 Model Performance Metrics"])

    # --- CLASSIFICATION TAB ---
    with tab1:
        st.header("Classify a New Signal")
        
        # --- Sidebar for User Input ---
        with st.sidebar:
            st.header("Signal Characteristics")
            input_data = {}
            for col in feature_columns:
                label = col.replace('_', ' ').replace('rearth', '(R_earth)').replace('teff', 'Teff (K)').title()
                default_val = float(df_full[col].median())
                min_val = float(df_full[col].min())
                max_val = float(df_full[col].max())
                input_data[col] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val)
        
        input_df = pd.DataFrame([input_data])
        st.write("Current Input Data:")
        st.dataframe(input_df)

        if st.button("Classify Signal"):
            # --- Ensemble Prediction ---
            st.subheader("Ensemble Model Prediction")
            prediction = pipeline.predict(input_df)[0]
            prediction_proba = pipeline.predict_proba(input_df)[0]
            
            result_class = disposition_map_inv[prediction]
            confidence = prediction_proba.max()
            
            col1, col2 = st.columns(2)
            col1.metric(label="Predicted Class", value=result_class)
            col2.metric(label="Model Confidence", value=f"{confidence:.2%}")
            
            # --- Individual Model Breakdown ---
            st.subheader("Individual Model Predictions (The 'Voters')")
            st.write("See how each model in the ensemble contributed to the final prediction.")
            
            scaler = pipeline.named_steps['scaler']
            input_scaled = scaler.transform(input_df)
            
            cols = st.columns(3)
            base_models = pipeline.named_steps['ensemble'].estimators_
            model_names = ['LightGBM', 'XGBoost', 'CatBoost']

            for i, model in enumerate(base_models):
                with cols[i]:
                    # --- THIS IS THE FIX ---
                    # Ensure the prediction is a standard Python integer
                    pred_ind = int(model.predict(input_scaled)[0])
                    prob_ind = model.predict_proba(input_scaled)[0]
                    
                    st.metric(label=f"{model_names[i]} Prediction", value=disposition_map_inv[pred_ind])
                    st.metric(label="Confidence", value=f"{prob_ind.max():.2%}")
                    
                    with st.expander("Show probabilities"):
                        prob_df = pd.DataFrame([prob_ind], columns=class_names)
                        st.dataframe(prob_df)

    # --- METRICS TAB ---
    with tab2:
        st.header("Model Performance on Held-Out Test Data")
        
        st.metric(label="Overall Accuracy", value=f"{metrics['accuracy']:.4f}")
        
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(metrics)
        with col2:
            st.subheader("Feature Importance")
            plot_feature_importance(metrics)