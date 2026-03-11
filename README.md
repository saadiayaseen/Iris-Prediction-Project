# 🌸 Iris Flower Prediction Web App

A **Machine Learning web application** built with **Python and Streamlit** that predicts the species of an Iris flower based on its physical measurements.

🔗 **Live App:**
https://iris-prediction-project-yjx4v3morekeudmrwmtzkg.streamlit.app/

---

# 📌 Project Overview

This project uses a trained **machine learning classification model** to identify the species of an Iris flower based on four features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The model predicts one of three species:

* **Iris Setosa**
* **Iris Versicolor**
* **Iris Virginica**

The famous **Iris dataset contains 150 samples across these three species and four measurements**, making it one of the most widely used datasets for teaching machine learning classification. ([GeeksforGeeks][1])

---

# 🚀 Features

### 🔹 AI Prediction

Users enter flower measurements and the model predicts the species instantly.

### 🔹 Confidence Score

Displays how confident the model is about the prediction.

### 🔹 Flower Image Display

Shows the predicted flower image after classification.

### 🔹 Species Comparison

Explains how the predicted flower differs from the other two species.

### 🔹 Dataset Visualization

Interactive scatter plots allow users to explore relationships between features in the Iris dataset.

### 🔹 Interactive UI

* Dark themed modern interface
* Smooth prediction animation
* Interactive charts and controls

---

# 🧠 Machine Learning Model

The model was trained using the **Iris dataset**, which contains:

* **150 samples**
* **3 flower species**
* **4 numerical features**

Features used:

| Feature      | Description               |
| ------------ | ------------------------- |
| Sepal Length | Length of the sepal in cm |
| Sepal Width  | Width of the sepal in cm  |
| Petal Length | Length of the petal in cm |
| Petal Width  | Width of the petal in cm  |

These measurements allow machine learning models to classify the species accurately. ([GeeksforGeeks][1])

---

# 🛠️ Tech Stack

**Frontend**

* Streamlit

**Backend**

* Python

**Machine Learning**

* Scikit-learn
* NumPy
* Pandas

**Model**

* Random Forest Classifier

---

# 📂 Project Structure

```
iris-prediction-project
│
├── app.py
├── iris_model.pkl
├── lable_encoder.pkl
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/iris-prediction-project.git
cd iris-prediction-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

# 🌐 Deployment

This application is deployed using **Streamlit Cloud**.

Live version:

https://iris-prediction-project-yjx4v3morekeudmrwmtzkg.streamlit.app/

---

# 📊 Example Prediction

Input values:

```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

Prediction:

```
Iris Setosa
Confidence: 98%
```

---

# 🎯 Learning Outcomes

This project demonstrates:

* Machine Learning Classification
* Model Deployment with Streamlit
* Interactive Data Visualization
* Building ML Web Applications
* UI Design for ML dashboards

---

# 👩‍💻 Author

**Saadia Yaseen Syed**

Machine Learning & Data Science Student

---

# 📜 License

This project is open-source and available for educational purposes.

[1]: https://www.geeksforgeeks.org/iris-dataset/?utm_source=chatgpt.com "Iris Dataset - GeeksforGeeks"
