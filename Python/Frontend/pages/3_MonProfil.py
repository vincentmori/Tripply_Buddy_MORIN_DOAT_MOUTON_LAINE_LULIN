import streamlit as st
from components.header import display_header
from components.footer import display_footer
from styles.load_css import load_css
from components.login_dialog import login_dialog

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------
def content_compte():
    st.title("Mon compte")

if st.session_state["STATUT_CONNEXION"]:
    content_compte()
else:
    login_dialog()

# -------------------------------
# FOOTER
# -------------------------------
display_footer()