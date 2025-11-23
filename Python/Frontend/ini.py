# Python/Frontend/init_session.py

import streamlit as st

def init_session_state():
    # Initialisation de l'état de connexion
    if 'STATUT_CONNEXION' not in st.session_state:
        st.session_state['STATUT_CONNEXION'] = False
        
    # Initialisation de l'identifiant utilisateur
    if 'username' not in st.session_state:
        st.session_state['username'] = None