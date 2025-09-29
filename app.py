import streamlit as st
import pandas as pd
import joblib

@st.cache_resource

def load_artifacts():
    model = joblib.load("models/cover_type_prediction_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model , scaler

model , scaler = load_artifacts()

numeric_features = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

categorical_features = [f"Wilderness_Area{i}" for i in range(1, 5)] + \
                       [f"Soil_Type{i}" for i in range(1, 41)]
all_features = numeric_features + categorical_features

st.set_page_config(page_title="Forest Cover Type Predictor", layout="wide")
st.title("🌲 Forest Cover Type Predictor")

user_data = {}
st.header("📥 Enter the input features")

st.subheader("Numeric Features")
for feature in numeric_features:
    user_data[feature] = st.number_input(feature, value=0.0)

st.subheader("Wilderness Area (One-hot Encoded)")
for feature in categorical_features[:4]:
    user_data[feature] = st.selectbox(feature, [0, 1], index=0)

st.subheader("Soil Type (One-hot Encoded)")
cols = st.columns(4)
for i, feature in enumerate(categorical_features[4:]):
    col_idx = i %4
    user_data[feature] = cols[col_idx].selectbox(feature, [0, 1], index=0)

input_df = pd.DataFrame([user_data])
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

if st.button("Predict Cover Type"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Forest Cover Type: {prediction[0]}")