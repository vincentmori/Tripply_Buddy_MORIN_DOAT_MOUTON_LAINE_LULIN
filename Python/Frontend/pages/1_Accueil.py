import streamlit as st
from components.header import display_header_with_dialog_trigger
from components.footer import display_footer
from styles.load_css import load_css

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header_with_dialog_trigger()

# -------------------------------
# CONTENT
# -------------------------------
def content_accueil():
    st.title("Bienvenue sur TripplyBuddy ✈️")
    st.write("Page d'accueil")
    
content_accueil()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()