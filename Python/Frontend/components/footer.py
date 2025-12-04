import streamlit as st

# ---------------------------
# PETIT FOOTER POUR LE SITE
# ---------------------------

def display_footer():
    st.markdown("""
    <div class="footer-text">
        © 2025 TripplyBuddy — Tous droits réservés 
    </div>
    """, unsafe_allow_html=True)