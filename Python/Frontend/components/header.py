import streamlit as st

# -----------------------
# FONCTION DE NAVIGATION
# -----------------------
def navigate_to(page_name):
    """Fonction pour changer de page tout en conservant l'état de la session."""
    st.switch_page(f"pages/{page_name}.py")


# ------------------------------------------------------
# HEADER AVEC LES BOUTONS DE NAVIGATION ENTRE LES PAGES
# ------------------------------------------------------
def display_header():
    header_container = st.container()
    
    with header_container:
        col1, col2, col3, col4 = st.columns([1, 0.2, 0.2, 0.2])
        
        with col1:
            st.markdown("""<div id="logo-page">TripplyBuddy</div>""", unsafe_allow_html=True)
            
        with col2:
            if st.button("Accueil", key="nav_home", use_container_width=True):
                navigate_to("1_Accueil")
        
        with col3:
            if st.button("Mon Profil", key="nav_profile", use_container_width=True):
                navigate_to("3_MonProfil")
                
        with col4:
            if st.button("À propos", key="nav_about", use_container_width=True):
                navigate_to("2_Apropos")
    
    st.markdown("---")