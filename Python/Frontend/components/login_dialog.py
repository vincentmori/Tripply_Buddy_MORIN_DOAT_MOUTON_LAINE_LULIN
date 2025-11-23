import streamlit as st

@st.dialog("Connexion")
def login_dialog():   
    # st.dialog utilise des widgets Streamlit normaux à l'intérieur
    username = st.text_input("Identifiant")
    password = st.text_input("Mot de passe", type='password')

    if st.button("Se connecter", use_container_width=True):
        if not username.strip() or not password.strip():
            st.error("Veuillez saisir vos identifiants.")
        else:
            st.session_state['STATUT_CONNEXION'] = True 
            st.session_state['username'] = username.strip() 
            st.success("Connexion réussie! Redémarrage de la page...")
            st.rerun()