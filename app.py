import streamlit as st
import joblib

# --------------------------
# Load trained model
# --------------------------
model = joblib.load("triage_model.pkl")

# --------------------------
# Streamlit UI
# --------------------------

st.title("🩺 Medical Triage Chat Assistant")
st.write("Enter your symptoms and the model will suggest the right doctor/specialty.")

# User Inputs
symptoms = st.text_area("Describe your symptoms:", height=120, placeholder="Example: Mujhe 2 din se fever aur cough ho raha hai")

age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender = st.selectbox("Gender", ["male", "female", "other"])

# Button
if st.button("Get Recommendation"):

    # Prepare input for model
    final_input = f"{symptoms} Age:{age} Gender:{gender}"

    # Prediction
    result = model.predict([final_input])[0]

    # Display
    st.success(f"🩻 Recommended Specialty: **{result}**")

    # Extra information
    st.info("⚠️ This is not a medical diagnosis. Always consult a certified doctor for accurate medical advice.")
