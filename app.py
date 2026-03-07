import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from io import BytesIO

# Imports pour le traitement de documents
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Imports pour le rapport PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- LOGIQUE TTU-MC3 ---

class TriadicProcessor:
    def __init__(self):
        # Modèle léger pour transformer le texte en vecteur
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pca = PCA(n_components=3)

    def get_signature(self, text):
        """Transforme le texte en coordonnées (M, C, D)"""
        embedding = self.model.encode([text])
        # Simulation d'une projection 3D (M, C, D)
        # Note : En pratique, on calibrerait la PCA sur un corpus
        sig = np.tanh(embedding[0][:3]) # Normalisation entre -1 et 1
        return sig

class InfiniteVTM:
    def lyapunov_dynamic(self, phi, t, target):
        m, c, d = phi
        tm, tc, td = target
        
        # Équations de la chute vers l'attracteur
        # La dissipation d pilote la vitesse de convergence
        dm = -0.5 * (m - tm) + 0.1 * c
        dc = (tc - c) * abs(d) - 0.1 * m
        dd = -0.2 * d + 0.05 * (c**2)
        return [dm, dc, dd]

    def simulate(self, target):
        t = np.linspace(0, 50, 1000)
        phi0 = [0.1, 0.0, 0.5] # État initial arbitraire
        traj = odeint(self.lyapunov_dynamic, phi0, t, args=(target,))
        return t, traj

# --- INTERFACE STREAMLIT ---

st.title("🌐 VTM : Plateforme de Calcul Triadique")
st.info("Ici, le document devient un paysage. Le calcul est sa stabilisation.")

uploaded_file = st.file_uploader("Charger un document", type=['pdf', 'docx', 'txt'])

if uploaded_file:
    # 1. Extraction (Simplifiée pour l'exemple)
    text = "Exemple de contenu extrait du document..." 
    st.success("Document ingéré par le champ de cohérence.")

    # 2. Analyse
    proc = TriadicProcessor()
    signature = proc.get_signature(text)
    
    vtm = InfiniteVTM()
    t, traj = vtm.simulate(signature)

    # 3. Visualisation
    st.subheader("📊 Convergence de la Triade (M, C, D)")
    fig, ax = plt.subplots()
    ax.plot(t, traj[:, 0], label="Mémoire (M)")
    ax.plot(t, traj[:, 1], label="Cohérence (C)")
    ax.plot(t, traj[:, 2], label="Dissipation (D)")
    ax.legend()
    st.pyplot(fig)

    # 4. Topologie
    st.subheader("🌀 Observation de l'Attracteur")
    # 
    # (Visualisation 2D simplifiée de l'espace des phases)
    fig2, ax2 = plt.subplots()
    ax2.plot(traj[:, 0], traj[:, 1], color='purple')
    ax2.scatter(signature[0], signature[1], color='red', label="Cible")
    ax2.set_xlabel("M")
    ax2.set_ylabel("C")
    st.pyplot(fig2)

    st.button("Générer le rapport PDF complet")
