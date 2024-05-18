import streamlit as st

def app():
    st.title("Models")
    st.write("Welcome to the Home Page!")

    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/linear_regression.py", label="Linear Regression", icon="1️⃣")


if __name__ == "__main__":
    app()
