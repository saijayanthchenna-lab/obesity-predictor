
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Obesity Predictor",
    page_icon="⚖️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "model (4).pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    encoder = joblib.load(BASE_DIR / "encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# Website heading
st.title("⚖️ Obesity Predictor")
st.write("Predict an obesity classification using machine learning.")

st.divider()
st.subheader("Enter Your Details")

with st.form("obesity_details"):
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)

    submitted = st.form_submit_button(
        "Predict Obesity Category 🔍",
        use_container_width=True
    )

if submitted:
    try:
        # Prepare numerical features in training order
        numeric_data = pd.DataFrame([{
            "Age": age,
            "Height": height,
            "Weight": weight
        }])

        # Prepare categorical feature
        categorical_data = pd.DataFrame([{
            "Gender": gender
        }])

        # Apply the fitted transformations
        scaled_values = scaler.transform(numeric_data)
        encoded_values = encoder.transform(categorical_data)

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=numeric_data.columns
        )

        encoded_df = pd.DataFrame(
            encoded_values,
            columns=encoder.get_feature_names_out(["Gender"])
        )

        input_data = pd.concat(
            [scaled_df, encoded_df],
            axis=1
        )

        # Predict
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric("Predicted Category", str(prediction))
        st.caption(
            "Educational estimate only; not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
