
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

# Model file path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model (4).pkl"

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Website heading
st.title("⚖️ Obesity Predictor")
st.write(
    "Predict an obesity classification using your "
    "trained machine learning model."
)

st.divider()

# Input form
st.subheader("Enter Your Details")

with st.form("obesity_details"):
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0
    )

    submitted = st.form_submit_button(
        "Predict Obesity Category 🔍",
        use_container_width=True
    )

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            "Predicted Category",
            str(prediction)
        )

        st.caption(
            "This is a machine learning estimate, "
            "not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
```
