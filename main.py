import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, TheilSenRegressor, HuberRegressor, PassiveAggressiveRegressor, ARDRegression, BayesianRidge, ElasticNet, OrthogonalMatchingPursuit
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

#JUDUL
st.title("Aplikasi Prediksi Dataset")

# UPLOAD DATASET
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    file_type = uploaded_file.type
    if file_type == "text/csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.header("Tabel Dataset (10 data teratas)")
    st.table(data.head(10))
    # st.write(file_type)

    st.header("Data Column")
    st.table(data.columns)

    st.header("Cek Data Kosong")
    st.table(data.isnull().any())
    
    st.header("Pilih kolom fitur (X)")
    options = st.multiselect(
    'Pilih kolom fitur (X)',data.columns)
    st.write('You selected:', options)
    data_X = data[options]
    X = data_X.values
    
    st.header("Kolom Fitur")
    st.table(data_X.head())

    st.divider()

    st.header("Pilih kolom target (Y)")
    options_target = st.multiselect(
    'Pilih kolom target (Y)',data.columns)
    st.write('You selected:', options_target)
    data_Y = data[options_target]

    # Split data into features (X) and target variable (y)
    y = data_Y.values
    st.header("Kolom Target")
    st.table(data_Y.head())

    # Split data into train, test sets
    st.divider()
    inputTestSize = 20
    inputTestSize = st.slider("Presentase data",1,100, 20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=inputTestSize/100, random_state=42)

    # Define models
    models = {
        'ExtraTrees': ExtraTreesRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'GaussianProcess': GaussianProcessRegressor(),
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Lars': Lars(),
        'TheilSen': TheilSenRegressor(),
        'Huber': HuberRegressor(),
        'PassiveAggressive': PassiveAggressiveRegressor(),
        'ARD': ARDRegression(),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(),
        'OMP': OrthogonalMatchingPursuit(),
        'SVR': SVR(),
        'NuSVR': NuSVR(),
        'LinearSVR': LinearSVR(),
        'KernelRidge': KernelRidge(),
        'RandomForest': RandomForestRegressor(),
        'KNeighbors': KNeighborsRegressor(),
        'PLSRegression': PLSRegression()
    }

    if options and options_target:
        # Visualize predictions for each model on train and test sets
        st.divider()
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            # Predictions on train set
            y_train_pred = model.predict(X_train)

            # Predictions on test set
            y_test_pred = model.predict(X_test)

            # Calculate RMSE and R-Square
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            r2_train = r2_score(y_train, y_train_pred)

            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)

            # Plotting
            plt.figure(figsize=(8, 6))

            plt.scatter(y_train, y_train_pred, label=f'Train ({model_name})\nRMSE: {rmse_train:.3f}, R-Square: {r2_train:.3f}', color='blue', alpha=0.7)
            plt.scatter(y_test, y_test_pred, label=f'Test ({model_name})\nRMSE: {rmse_test:.3f}, R-Square: {r2_test:.3f}', color='red', alpha=0.7)

            plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='gray', label='Ideal Line (Train)')
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black', label='Ideal Line (Test)')

            plt.title(f'Train and Test Set - {model_name}')
            plt.xlabel('Actual Target Variable')
            plt.ylabel('Predicted Target Variable')
            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)

