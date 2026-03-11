import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
from sklearn.datasets import load_iris

# Load model
model = joblib.load("iris_model.pkl")
encoder = joblib.load("lable_encoder.pkl")

st.set_page_config(page_title="Iris AI Detector", page_icon="🌸", layout="wide")

# -------- DARK UI --------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#020617,#0f172a);
color:white;
}

.title{
text-align:center;
font-size:48px;
font-weight:700;
color:#f1f5f9;
}

.subtitle{
text-align:center;
font-size:18px;
color:#94a3b8;
}

.main-card{
background: rgba(255,255,255,0.05);
backdrop-filter: blur(15px);
padding:35px;
border-radius:20px;
border:1px solid rgba(255,255,255,0.1);
box-shadow:0 10px 35px rgba(0,0,0,0.6);
}

.prediction-box{
background: rgba(59,130,246,0.15);
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:600;
animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
from {opacity:0; transform: translateY(10px);}
to {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown('<p class="title">🌸 Iris Flower Prediction Project </p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Developed by Saadia Yaseen Syed</p>', unsafe_allow_html=True)

# -------- SIDEBAR --------
st.sidebar.header("Options")

show_conf = st.sidebar.checkbox("Show Confidence Score", True)
show_prob = st.sidebar.checkbox("Show Probability Chart")
show_dataset = st.sidebar.checkbox("Show Dataset Visualization")

# -------- INPUT CARD --------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.subheader("Enter Flower Measurements")

col1,col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)",4.0,8.0,5.1)
    sepal_width = st.slider("Sepal Width (cm)",2.0,4.5,3.5)

with col2:
    petal_length = st.slider("Petal Length (cm)",1.0,7.0,1.4)
    petal_width = st.slider("Petal Width (cm)",0.1,2.5,0.2)

input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

# -------- PREDICTION --------
if st.button("Run AI Prediction 🚀"):

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)

    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)

    flower = encoder.inverse_transform(prediction)[0]
    confidence = np.max(probs)*100

    st.markdown(
        f'<div class="prediction-box">Prediction: {flower}</div>',
        unsafe_allow_html=True
    )

    if show_conf:
        st.success(f"Confidence Score: {confidence:.2f}%")

    if show_prob:
        df = pd.DataFrame(probs,columns=encoder.classes_)
        st.bar_chart(df)

# -------- FLOWER IMAGE --------

    st.subheader("Predicted Flower")

    if flower == "Iris-setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg")

    elif flower == "Iris-versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg")

    elif flower == "Iris-virginica":
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg")

# -------- DIFFERENCE SECTION --------

    st.write("---")
    st.subheader("How This Flower Differs From the Others")

    if flower == "Iris-setosa":
        st.info("""
**Iris Setosa** is different from the other two species because:

• It has the **smallest petals** among the three species  
• Petal length is usually **less than 2 cm**  
• It is the **easiest species for ML models to classify** because it is clearly separated in the dataset
""")

    elif flower == "Iris-versicolor":
        st.info("""
**Iris Versicolor** differs because:

• It has **medium-sized petals**  
• Petal length typically ranges **between 3–5 cm**  
• It overlaps slightly with Virginica, making it **harder for models to distinguish**
""")

    elif flower == "Iris-virginica":
        st.info("""
**Iris Virginica** differs because:

• It has the **largest petals** of the three species  
• Petal length is typically **greater than 5 cm**  
• It is often **confused with Versicolor**, but its measurements are usually larger
""")

st.markdown('</div>', unsafe_allow_html=True)

# -------- DATASET VISUALIZATION --------

if show_dataset:

    st.write("---")
    st.header("Iris Dataset Visualization")

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[
            "sepal length",
            "sepal width",
            "petal length",
            "petal width"
        ]
    )

    df["species"] = iris.target

    x_feature = st.selectbox("Select X Axis",df.columns[:-1])
    y_feature = st.selectbox("Select Y Axis",df.columns[:-1],index=1)

    st.scatter_chart(df,x=x_feature,y=y_feature)

st.write("---")
st.caption("Machine Learning • Streamlit • Iris Dataset")