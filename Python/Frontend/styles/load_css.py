import streamlit as st
from pathlib import Path

def load_css():
    """
    Load and 
    Charge et injecte le contenu d'un fichier CSS en utilisant un chemin absolu relatif.
    """
    #  Obtenir le chemin du répertoire du script Python en cours d'exécution
    current_dir = Path(__file__).parent 
    
    # Construire le chemin complet vers styles.css
    css_file_path = current_dir / "styles.css"
    
    try:
        # Ouvrir le fichier et injecter le contenu
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Afficher une erreur claire si le fichier n'est pas trouvé
        st.error(f"Erreur: Le fichier CSS n'a pas été trouvé à l'emplacement : {css_file_path}")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement du CSS : {e}")