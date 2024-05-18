import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "About"])

if page == "Home":
    import home
    home.app()
elif page == "About":
    import about
    about.app()