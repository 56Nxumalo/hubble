import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page title and layout
st.set_page_config(page_title="Hubble Constant Estimation", layout="wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Introduction", "Data Explorer", "Model Training", "Results & Analysis", "Contact"],
)

# Dummy dataset (replace with actual data)
dummy_data = pd.DataFrame({
    "Redshift (z)": np.linspace(0.01, 2, 100),
    "Magnitude (m)": np.random.normal(19, 0.5, 100),
    "Error in m": np.random.normal(0.1, 0.02, 100),
})

# Sections based on menu selection
if menu == "Introduction":
    st.title("Hubble Constant Estimation Using Machine Learning")
    st.write("""
        This application showcases the use of machine learning, specifically Support Vector Machines (SVM),
        to estimate the Hubble constant based on astrophysical data. 
        Explore the dataset, train models, and analyze results in the respective sections.
    """)

elif menu == "Data Explorer":
    st.title("Data Explorer")
    st.write("### Dataset Preview")
    st.dataframe(dummy_data)

    # Visualization options
    st.sidebar.header("Visualization Options")
    x_axis = st.sidebar.selectbox("X-Axis", dummy_data.columns)
    y_axis = st.sidebar.selectbox("Y-Axis", dummy_data.columns)
    
    st.write("### Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(dummy_data[x_axis], dummy_data[y_axis], color='blue', alpha=0.7)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{y_axis} vs {x_axis}")
    st.pyplot(fig)

elif menu == "Model Training":
    st.title("Model Training")
    st.write("### Train a Support Vector Regressor (SVR) Model")
    
    # Splitting the data
    X = dummy_data[["Redshift (z)", "Magnitude (m)"]]
    y = dummy_data["Redshift (z)"] * 70 + np.random.normal(0, 5, len(dummy_data))  # Dummy H(z) values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model parameters
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
    C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0)
    
    # Train the model
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display performance metrics
    st.write("### Performance Metrics")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

elif menu == "Results & Analysis":
    st.title("Results & Analysis")
    st.write("### Predicted vs Actual H(z)")
    
    # Check if y_test and y_pred are defined
    if 'y_test' in locals() and 'y_pred' in locals():
        predictions = pd.DataFrame({
            "True H(z)": y_test,
            "Predicted H(z)": y_pred,
        })

        st.dataframe(predictions)

        # Plot True vs Predicted H(z)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color='green')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("True H(z)")
        ax.set_ylabel("Predicted H(z)")
        ax.set_title("True vs Predicted H(z)")
        st.pyplot(fig)
    else:
        st.write("Model not trained yet. Please train the model first.")

elif menu == "Contact":
    st.title("Contact")
    st.write("For any questions or collaborations, please contact:")
    st.write("**Name:** Sinenhlanhla Mbali Nxumalo")
    st.write("**Email:** snehnxumalo56@gmail.com")
    st.write("**Institution:** University of Zululand")

