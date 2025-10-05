import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import os

# --- Import all necessary model training libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np


# ==============================================================================
#  STEP 1: DATA PREPARATION LOGIC (from data_preparation.py)
# ==============================================================================
INPUT_FILE = 'df_final_cleaned.csv'
PREPARED_DATA_FILE = 'exoplanet_data_merged_for_ensemble.csv'

def prepare_merged_data_for_ensemble():
    st.write(f"Loading raw data from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    
    feature_columns = ['period_days', 'duration_hours', 'depth_ppm', 'planet_radius_rearth', 'stellar_teff', 'stellar_radius_solar', 'stellar_logg']
    target_column = 'disposition'
    columns_to_keep = feature_columns + [target_column]

    df_clean = df[columns_to_keep].copy()
    initial_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    st.write(f"Dropped {initial_rows - len(df_clean)} rows with missing values.")

    disposition_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
    df_clean = df_clean[df_clean['disposition'].isin(disposition_map.keys())]
    df_clean['disposition'] = df_clean['disposition'].map(disposition_map)
    
    df_clean.to_csv(PREPARED_DATA_FILE, index=False)
    st.write(f"Cleaned data saved to '{PREPARED_DATA_FILE}'.")

# ==============================================================================
#  STEP 2: MODEL TRAINING LOGIC (from train_model.py)
# ==============================================================================
def train_ensemble_model():
    st.write(f"Loading prepared data from '{PREPARED_DATA_FILE}'...")
    df = pd.read_csv(PREPARED_DATA_FILE)

    X = df.drop('disposition', axis=1)
    y = df['disposition']
    feature_columns = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf1 = LGBMClassifier(random_state=42)
    clf2 = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    clf3 = CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass')

    eclf1 = VotingClassifier(estimators=[('lgbm', clf1), ('xgb', clf2), ('catboost', clf3)], voting='soft')
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('ensemble', eclf1)
    ])

    st.write("Training the full pipeline... This is the long step.")
    pipeline.fit(X_train, y_train)
    st.write("Training complete!")

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['False Positive', 'Candidate', 'Confirmed'], output_dict=True)
    base_models = pipeline.named_steps['ensemble'].estimators_
    importances = [model.feature_importances_ for model in base_models]
    avg_importances = np.mean(importances, axis=0)
    feature_importance_data = pd.DataFrame({'feature': feature_columns, 'importance': avg_importances}).sort_values(by='importance', ascending=False)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': report,
        'y_test': y_test.to_numpy(),
        'y_pred': y_pred,
        'feature_importances': feature_importance_data
    }
    
    joblib.dump(pipeline, 'exoplanet_ensemble_model.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    joblib.dump(metrics, 'training_metrics.joblib')
    st.write("Model and metric files saved to disk.")

# ==============================================================================
#  STEP 3: MAIN APP LOGIC
# ==============================================================================
# --- Page Configuration ---
st.set_page_config(page_title="Exoplanet Classification", page_icon="🛰️", layout="wide")

# --- Check if model exists, if not, run setup ---
MODEL_FILE = 'exoplanet_ensemble_model.joblib'
if not os.path.exists(MODEL_FILE):
    st.warning("Performing first-time setup. This will take several minutes.")
    prepare_merged_data_for_ensemble()
    train_ensemble_model()
    st.success("Setup complete! The app is now ready.")
    st.balloons()


# --- Load Assets (now guaranteed to exist) ---
@st.cache_resource
def load_assets():
    pipeline = joblib.load('exoplanet_ensemble_model.joblib')
    columns = joblib.load('feature_columns.joblib')
    metrics = joblib.load('training_metrics.joblib')
    df_full = pd.read_csv(PREPARED_DATA_FILE)
    return pipeline, columns, df_full, metrics

pipeline, feature_columns, df_full, metrics = load_assets()
disposition_map_inv = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
class_names = list(disposition_map_inv.values())

# --- Helper Functions for Plotting ---
def plot_confusion_matrix(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(metrics['y_test'], metrics['y_pred'], display_labels=class_names, cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix on Test Set")
    st.pyplot(fig)

def plot_feature_importance(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=metrics['feature_importances'], palette='viridis', ax=ax)
    ax.set_title("Average Feature Importance Across Ensemble")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


# --- UI Layout ---
st.title("Exoplanetary classification: An ensamble model prediction approach  🛰️ 🪐 ")
st.write("""
This tool uses a high-performance Ensemble Model trained on a merged dataset from the Kepler, K2, and TESS missions. 
Use the tabs below to perform live classifications by changing the hyperparameters from the left tab and view the accuracy and confidence each model contributed to the ensemble outcome. Explore the model's performance metrics from its last training run and get more context from the project overview tab.
""")
# ... (The rest of your UI code is identical and does not need to be changed)
# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Live Classification", "📈 Model Performance", "📖 Project Overview"])
# ... (pasting the rest of your UI code here)
with tab1:
    st.header("Classify a New Signal")
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
        st.subheader("Ensemble Model Prediction")
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0]
        result_class = disposition_map_inv[prediction]
        confidence = prediction_proba.max()
        col1, col2 = st.columns(2)
        col1.metric(label="Predicted Class", value=result_class)
        col2.metric(label="Model Confidence", value=f"{confidence:.2%}")
        st.subheader("Individual Model Predictions (The 'Voters')")
        st.write("See how each model in the ensemble contributed to the final prediction.")
        scaler = pipeline.named_steps['scaler']
        input_scaled = scaler.transform(input_df)
        cols = st.columns(3)
        base_models = pipeline.named_steps['ensemble'].estimators_
        model_names = ['LightGBM', 'XGBoost', 'CatBoost']
        for i, model in enumerate(base_models):
            with cols[i]:
                pred_ind = int(model.predict(input_scaled)[0])
                prob_ind = model.predict_proba(input_scaled)[0]
                st.metric(label=f"{model_names[i]} Prediction", value=disposition_map_inv[pred_ind])
                st.metric(label="Confidence", value=f"{prob_ind.max():.2%}")
                with st.expander("Show probabilities"):
                    prob_df = pd.DataFrame([prob_ind], columns=class_names)
                    st.dataframe(prob_df)
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

    with tab3:
        
        st.subheader("Automating Discovery")
        st.markdown("""
        Space telescopes missions have generated a massive amount of data throughout the years, identifying thousands of potential exoplanets. Historically, vetting these candidates to distinguish real planets from instrumental noise or other astrophysical phenomena (like binary stars) has been a manual, time-consuming process.

        The goal of this project is to provide a reliable Machine Learning classification and prediction model that can enable faster and more efficient analysis of new and existing astronomical data while remaining open and truthful about its own confidence and accuracy.
        """)

        st.subheader("A Multi-Mission Approach")
        st.markdown("""
        This model was trained on a merged dataset combining publicly available data from three key NASA missions:
        *   **Kepler:** The pioneering mission that provided a vast, deep survey of one patch of the sky.
        *   **K2:** The repurposed Kepler mission, which surveyed different areas of the sky along the ecliptic.
        *   **TESS:** The current-generation all-sky survey mission.

        By combining these sources, the model learns from a wider variety of stars and signal types, making it more robust and general-purpose than a model trained on a single mission's data. Aditionally, the model combines characteristics present in all three missions, such as transit features and stellar properties, to try to guarantee a confident outcome.
        """)

        st.subheader("The Variables That Matter")
        st.markdown("Recognizing how critical feature selection is, the model takes a set of variables that have direct physical significance to the transit method of exoplanet detection.")
        
        st.markdown("#### Transit Signature")
        st.markdown("""
        *   `period_days`: **Orbital Period.** The time it takes for the planet to complete one orbit. A consistent repeating period is the strongest evidence of an orbiting body.
        *   `duration_hours`: **Transit Duration.** How long the dimming event lasts. This must be physically plausible for a planet crossing its star.
        *   `depth_ppm`: **Transit Depth.** The percentage of starlight blocked. This is directly related to the planet's size relative to its star.
        """)

        st.markdown("#### Inferred and Stellar Properties")
        st.markdown("""
        *   `planet_radius_rearth`: **Planetary Radius.** The calculated size of the planet candidate (in Earth radii). This is a powerful feature for distinguishing between planet-sized objects and much larger stars or brown dwarfs.
        *   `stellar_teff`: **Stellar Effective Temperature.** The temperature of the host star. This provides context about the type of star, which influences the expected signal and noise characteristics.
        *   `stellar_radius_solar`: **Stellar Radius.** The size of the host star. This is critically important as it's needed to convert the relative transit depth into an absolute planetary radius.
        *   `stellar_logg`: **Stellar Surface Gravity.** This helps differentiate between main-sequence (dwarf) stars and evolved giant stars, which have very different properties and are less likely to host detectable transiting planets.
        """)
        
        st.subheader("The Ensemble")
        st.markdown("""
        Instead of relying on a single model, this project uses a "Voting Ensemble", which combines the predictions of three top-tier gradient boosting models: LightGBM, XGBoost, and CatBoost. This approach leverages the strengths of each individual model to create a more accurate and robust overall classifier. The idea was adapted from Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification. Electronics, 13(19), 3950.
        
                    
        This ensemble uses 'soft' voting. Rather than a simple majority vote, it averages the confidence scores (probabilities) from each of the three models to make a final, nuanced decision. This is a state-of-the-art technique for achieving high performance in classification tasks.
        """)
