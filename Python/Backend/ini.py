import streamlit as st
import pandas as pd
from Python.Backend.recup_data import recup_travel, recup_users
from Python.Backend.genV2 import DESTINATIONS
from Model.predict import get_recommendation

def chargement_df():
    # Charger les deux datasets avec les fonctions directement depuis la bdd sur le cloud
    df_users = recup_users()
    df_connexion_users = df_users[["traveler_user_id", "mot_de_passe"]]

    df_travel = recup_travel()

    df_destinations = pd.DataFrame(DESTINATIONS)
    
    return {
        "df_users": df_users, 
        "df_connexion_users": df_connexion_users, 
        "df_destinations":df_destinations, 
        "df_travel": df_travel
        }

def init_session_state():
    df = chargement_df()
    
    if 'premier_affichage' not in st.session_state:
        st.session_state['premier_affichage'] = True
    
    # Initialisation de l'état de connexion
    if 'STATUT_CONNEXION' not in st.session_state:
        st.session_state['STATUT_CONNEXION'] = False
    
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
        
    if 'df_users' not in st.session_state:
        st.session_state['df_users'] = df["df_users"]
        
    # Initialisation du dataframe pour avoir les user_id et leur mot de passe associé
    if 'df_connexion_users' not in st.session_state:
        st.session_state['df_connexion_users'] = df["df_connexion_users"]
        
    if 'df_travel' not in st.session_state:
        st.session_state['df_travel'] = df["df_travel"]
        
    # Récupère les déstinations utilisés pour générer le dataset
    if 'df_destinations' not in st.session_state:
        st.session_state['df_destinations'] = df["df_destinations"]
        
    # Initialisation de l'utilisateur vide
    if 'user' not in st.session_state:
        st.session_state['user'] = None
        
    # Chemin vers un github public contenant les images associés a leur ville
    if 'chemin_image' not in st.session_state:
        st.session_state['chemin_image'] = "https://raw.githubusercontent.com/vincentmori/Image_ville_projet_rec_sys/refs/heads/main/"
        
def init_user(user_id):
    # Initialisation complete de l'utilisateur après la connexion
    st.session_state['STATUT_CONNEXION'] = True 
    
    df_users = st.session_state['df_users']
    
    mask_user = df_users["traveler_user_id"] == user_id
    
    user = df_users[mask_user]
    
    # Récupère les infromations de l'utilisateur
    st.session_state["user"] = user
    
    df_travel = st.session_state['df_travel']
    
    mask_histo_user = df_travel["User ID"] == user_id
    
    historique_user = df_travel[mask_histo_user]
    
    # Reset_index pour le tri réalisé ensuite
    historique_user = historique_user.reset_index(drop=True)
    
    historique_user["Start date"] = pd.to_datetime(historique_user["Start date"])
    
    # Récupère son hitsorique du plus récent au plus ancien 
    st.session_state["historique_user"] =  historique_user.sort_values(
        by="Start date",
        ascending=False
    ).reset_index(drop=True)

    # Récupère les recommandations de l'utilisateur en appelant la fonction reco_user qui appelle la fonction de prédiction 
    reco_user(user_id)
    
def reco_user(user_id):
    reco_user = get_recommendation(user_id)

    st.session_state["reco_user"] = reco_user
    