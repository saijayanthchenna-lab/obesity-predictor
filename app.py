
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Obesity Category Predictor",
    page_icon="⚖️",
    layout="centered"
)

# Load the trained Random Forest pipeline
@st.cache_resource
def load_model():
    return joblib.load("model (4).pkl")

model = load_model()

# Header
st.title("⚖️ Obesity Category Predictor")
st.write(
    "Enter the individual's details to predict "
    "the obesity category using Machine Learning."
)

st.info(
    "This is an educational ML prediction, "
    "not a medical diagnosis."
)

st.divider()

# User inputs
st.subheader("Enter Personal Details")

with st.form("prediction_form"):

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=1.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        max_value=300.0,
        value=65.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "Predict Category",
        use_container_width=True
    )

# Prediction
if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Height": [height],
        "Weight": [weight]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.success(f"Predicted Category: {prediction}")

        # Show class probabilities if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_df = pd.DataFrame({
                "Category": classes,
                "Model Score (%)": probabilities * 100
            }).sort_values(
                "Model Score (%)",
                ascending=False
            )

            st.subheader("Model Probability Scores")

            st.bar_chart(
                probability_df.set_index("Category")[
                    "Model Score (%)"
                ]
            )

            st.caption(
                "These scores are model-estimated probabilities, "
                "not clinical risk estimates."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
