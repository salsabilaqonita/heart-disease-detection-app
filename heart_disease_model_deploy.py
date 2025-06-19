import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

##-----------------------------------------------------------------------------------------------------------------------------------------------------##

import numpy as np
# import pandas as pd
import math
import random
import seaborn as sns
from scipy.stats import pearsonr, jarque_bera
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
import sklearn.metrics._scorer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

##-----------------------------------------------------------------------------------------------------------------------------------------------------##

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to heart disease machine learning dashboard

This dashboard created by : [@SalsabilaQonita](https://www.linkedin.com/in/salsabilaqonitakaltsum/)

**Disclaimer**:
- This app is for educational purposes only
- This app is not for diagnostical purposes
- Professional assistance is recommended to use this app
""")


def heart():
    st.write("""
    This app predicts the **Heart Disease**

    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1, 4, 2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write(
                "Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            thalach = st.sidebar.slider(
                "Detak jantung maksimum saat beraktivitas berat atau saat olahraga, dsb (bpm)", 71, 202, 80)
            trestbps = st.sidebar.slider(
                "Tekanan darah saat istirahat (mmHg)", min_value=60, max_value=200, step=20)
            chol = st.sidebar.slider(
                "Serum kolesterol", min_value=100, max_value=400, step=1)
            fbs = st.sidebar.slider(
                "Kadar gula darah saat puasa/belum makan (0 : <120 mg/dl, 1 : >120 mg/dl)", 0, 1, 1)
            restecg = st.sidebar.slider(
                "Hasil EKG saat istirahat dalam kategori (0 : Normal, 1 : ST-T, 2 : Hipertropi ventrikel kiri)", 0, 2, 1)
            slope = st.sidebar.slider(
                "Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider(
                "Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider(
                "Exercise induced angina (0 : 'No', 1 : 'Yes')", 0, 1, 1)
            ca = st.sidebar.slider("Jumlah pembuluh darah utama", 0, 3, 1)
            thal = st.sidebar.slider(
                "Hasil tes thalium (1 : Normal, 2 : Defek tetap, 3 : Defek dapat dipulihkan)", 1, 3, 1)
            sex = st.sidebar.selectbox(
                "Jenis Kelamin", ('Perempuan', 'Laki-laki'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1
            age = st.sidebar.slider("Usia", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': fbs,
                    'restecg': restecg,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

    input_df = user_input_features()
    img = Image.open("heart_img.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("best_model_rf.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['No Heart Disease' if prediction ==
                  0 else 'Yes, Potential Heart Disease, consider further professional assistance and diagnosis']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction result: {output}")

if __name__ == '__main__':
    heart()
