import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Prediction Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Custom CSS Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_xgboost_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


# Title Header
st.title("⚡ Predictive Analytics Dashboard")
st.markdown(
    "Provide customer/user attributes below to get real-time XGBoost predictions."
)
st.markdown("---")

# Input Form in Sidebar / Main Layout
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Input Parameters")

    age = st.slider("Age", min_value=18, max_value=100, value=35)

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
    )

    region = st.selectbox(
        "Region Code",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Region {x}",
    )

    occupation = st.selectbox(
        "Occupation Code",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: f"Type {x}",
    )

    income = st.number_input(
        "Income ($)", min_value=0, max_value=500000, value=50000, step=1000
    )

    predict_btn = st.button("🚀 Generate Prediction")

with col_result:
    st.subheader("🎯 Prediction Output")

    if predict_btn:
        try:
            model = load_xgboost_model()
            features = np.array([[age, gender, region, occupation, income]])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            conf_score = probabilities[prediction] * 100

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.success(f"### Result: Class 1 (Positive Outcome)")
            else:
                st.info(f"### Result: Class 0 (Negative Outcome)")

            st.metric(label="Model Confidence Score", value=f"{conf_score:.2f}%")
            st.progress(float(probabilities[1]))

            st.write("**Probability Breakdown:**")
            st.json(
                {
                    "Class 0 Probability": f"{probabilities[0]:.4f}",
                    "Class 1 Probability": f"{probabilities[1]:.4f}",
                }
            )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to process prediction: {e}")
    else:
        st.info("Adjust the values on the left and click **Generate Prediction**.")
