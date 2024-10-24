import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

# Title and description
st.title('Classifying Iris Flowers')
st.markdown('A toy model to classify iris flowers into Setosa, Versicolor, or Virginica')
# Section header
st.header("Plant Features")
# Create two columns for the input sliders
col1, col2 = st.columns(2)

# Sepal characteristics in the first column
with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

# Petal characteristics in the second column
with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)



# Button to trigger prediction
if st.button("Predict type of Iris"):
    # Assuming the predict function takes the four feature inputs and returns a prediction
    data = np.array([sepal_l, sepal_w, petal_l, petal_w]).reshape(1,-1)
    result = predict(data)
    st.write(f"The predicted type of Iris is: {result}")
