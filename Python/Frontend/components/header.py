import streamlit as st
from .login_dialog import login_dialog

def display_header():
    st.markdown("""
    <div id="custom-header">
        <div id="logo">TripplyBuddy</div>
        <div class="header-menu">
            <a href="/Accueil" target="_self">Accueil</a>
            <a href="/Apropos" target="_self">À propos</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_header_with_dialog_trigger():
    
    # Injecter le CSS (cela ne devrait pas être ici, mais si c'est nécessaire pour le debugging...)
    # Assurez-vous que load_css() est appelé au début du script!

    # 1. Utiliser un st.container pour définir la zone du header
    # Le style de fond et la position sticky doivent être appliqués à ce conteneur parent 
    # via les surcharges CSS vues précédemment.
    with st.container(border=False): 
        
        # 2. Créer les colonnes pour aligner le logo (à gauche) et le menu (à droite)
        # Ratio : (Petit Logo) : (Grand Espace Vide) : (Petit Menu)
        col_logo, col_spacer, col_nav = st.columns([1, 4, 2])
        
        # --- COLONNE LOGO ---
        with col_logo:
            # Injecte le HTML stylisé pour le logo
            st.markdown('<div id="logo">TripplyBuddy</div>', unsafe_allow_html=True)
            
        # --- COLONNE MENU DE NAVIGATION ---
        with col_nav:
            # 3. Utiliser des colonnes À L'INTÉRIEUR de col_nav pour aligner les 3 éléments du menu (Accueil, Connexion, À propos)
            col_home, col_login, col_about = st.columns([1, 1.2, 1])
            
            # --- LIEN ACCUEIL ---
            with col_home:
                st.markdown('<a href="/Accueil" target="_self" class="header-link">Accueil</a>', unsafe_allow_html=True)
            
            # --- BOUTON CONNEXION (Trigger de Dialogue) ---
            with col_login:
                # Appliquer la classe CSS pour que le bouton ressemble à un lien
                st.markdown('<div class="header-link-button">', unsafe_allow_html=True)
                if st.button("Connexion", key="header_login_trigger", use_container_width=True):
                    login_dialog()
                st.markdown('</div>', unsafe_allow_html=True)
                
            # --- LIEN À PROPOS ---
            with col_about:
                st.markdown('<a href="/Apropos" target="_self" class="header-link">À propos</a>', unsafe_allow_html=True)