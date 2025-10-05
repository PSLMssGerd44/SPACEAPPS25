import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Exoplanet Identifier AI (Ensemble)",
    page_icon="🔭",
    layout="wide"
)

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    """Load the trained ensemble pipeline and feature columns."""
    try:
        pipeline = joblib.load('exoplanet_ensemble_model.joblib')
        columns = joblib.load('feature_columns.joblib')
        df_full = pd.read_csv('exoplanet_data_merged_for_ensemble.csv')
        return pipeline, columns, df_full
    except FileNotFoundError:
        st.error("Model assets not found. Please run `data_preparation.py` and `train_model.py` first.")
        return None, None, None

pipeline, feature_columns, df_full = load_assets()
# Updated map for the 3-class classification
disposition_map_inv = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}

# --- UI Layout ---
st.title("🔭 General-Purpose Exoplanet Identifier")
st.write("""
This tool uses a high-performance **Ensemble Model** (LGBM, XGBoost, CatBoost) trained on a merged dataset from the **Kepler, K2, and TESS missions**. 
It can classify new signals as Confirmed, Candidate, or False Positive.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Classify a New Signal")
st.sidebar.write("Enter the signal characteristics.")

input_df = None
if feature_columns:
    input_data = {}
    for col in feature_columns:
        # Use a more descriptive label for the sidebar
        label = col.replace('_', ' ').replace('rearth', '(R_earth)').replace('teff', 'Teff (K)').title()
        default_val = float(df_full[col].median())
        min_val = float(df_full[col].min())
        max_val = float(df_full[col].max())
        input_data[col] = st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=default_val)
    
    input_df = pd.DataFrame([input_data])

# --- Main Content Area ---
st.header("Classification Result")
if input_df is not None:
    st.write("Input Data:")
    st.dataframe(input_df)

    if st.button("Classify with Ensemble Model"):
        with st.spinner("Analyzing signal with advanced models..."):
            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)
            
            result_class = disposition_map_inv[prediction[0]]
            confidence = prediction_proba[0].max()

            st.success(f"Classification Complete!")
            st.metric(label="Predicted Class", value=result_class)
            st.metric(label="Model Confidence", value=f"{confidence:.2%}")
            
            st.write("Full Prediction Probabilities:")
            proba_df = pd.DataFrame([prediction_proba[0]], columns=disposition_map_inv.values())
            st.dataframe(proba_df)
else:
    st.info("App is ready. Enter data in the sidebar to get a classification.")