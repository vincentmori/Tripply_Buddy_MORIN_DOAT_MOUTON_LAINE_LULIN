import streamlit as st
from components.header import display_header
from styles.load_css import load_css
from ini import init_session_state

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# INITIALIZATION SESSION STATE
# -------------------------------
init_session_state()

load_css()

# Redirection vers la page Accueil
st.switch_page("pages/1_Accueil.py")