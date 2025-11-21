import streamlit as st
from styles.load_css import load_css
from components.header import display_header
from components.footer import display_footer

st.set_page_config(page_title="TripplyBuddy", page_icon="🌍", layout="wide")

# -------------------------------
# CHARGEMENT DU CSS & HEADER
# -------------------------------
load_css()
display_header()

# -------------------------------
# CONTENT
# -------------------------------
def content_Apropos():

    # Définir la balise HTML pour le logo 
    logo_html_inline = '<span id="logo-page">TripplyBuddy</span>'

    st.title("About the Project 🎓")

    st.markdown(f"""
    Bienvenue sur {logo_html_inline} ! Ce projet a été réalisé dans le cadre d'un projet étudiant 
    sur les systèmes de recommandation personnalisée dans le domaine du voyage.
    """, unsafe_allow_html=True) 

    st.markdown('<div class="section-title">Objectif</div>', unsafe_allow_html=True)
    st.write("""
    Fournir des recommandations de destinations, hôtels et activités adaptées aux préférences des utilisateurs.
    """)

content_Apropos()
# -------------------------------
# FOOTER
# -------------------------------

display_footer()