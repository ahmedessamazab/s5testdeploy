import streamlit as st
import pandas as pd

st.title("Ahmed Azab portfolio")

st.image(
    "./images/cat.jpg",
    alt="Sunrise over a mountain ridge",
)

st.write("Welcome to my portfolio!")


name = st.text_input("Enter your name:")
age = st.slider("Select your age:", 0, 100, 25)

if st.button("Submit"):
    st.write(f"Hello {name}, you are {age} years old!")


df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})


st.dataframe(df)
