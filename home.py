import streamlit as st

def app():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/models.py", label="Models")
    st.page_link("pages/models.py", label="Page 2", disabled=True)
    st.page_link("http://www.google.com", label="Google", icon="🌎")


if __name__ == "__main__":
    app()
