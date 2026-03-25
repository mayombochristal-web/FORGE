import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from io import BytesIO
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import tempfile

# ============================================================================
# MODULE 1 : MOTEUR SÉMANTIQUE (TTU Encoder)
# ============================================================================
@st.cache_resource
def load_model():
    """Charge le modèle SentenceTransformer une seule fois."""
    return SentenceTransformer('all-MiniLM-L6-v2')

class TTUProcessor:
    def __init__(self):
        self.model = load_model()

    def extract_text(self, uploaded_file):
        """Extrait le texte d'un fichier PDF, DOCX ou TXT."""
        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = str(uploaded_file.read(), "utf-8")
        return text

    def compute_signature(self, text):
        """
        Projette le contenu sémantique sur les axes triadiques.
        Utilise des segments fixes de l'embedding pour M, C, D.
        """
        emb = self.model.encode([text])[0]
        # Découpage en trois parties égales (embedding de 384 dimensions)
        dim = len(emb)
        third = dim // 3
        m = np.mean(emb[:third])
        c = np.mean(emb[third:2*third])
        d = np.mean(emb[2*third:])
        # Normalisation dans [-1, 1]
        return np.tanh([m * 5, c * 5, d * 5])   # facteur 5 pour étendre la plage


# ============================================================================
# MODULE 2 : MOTEUR DYNAMIQUE (VTM Core)
# ============================================================================
class VTM_Core:
    def __init__(self, target, params=None):
        """
        target : tuple (M_target, C_target, D_target)
        params : dict avec les coefficients du système TTU canonique
        """
        self.target = target
        # Paramètres par défaut (issus de la forme normale TTU-MC³)
        self.params = {
            'alpha': 0.6,   # rappel mémoire
            'beta': 0.2,    # couplage M ← C
            'gamma': 0.5,   # rappel cohérence
            'delta': 0.3,   # couplage C ← M
            'epsilon': 0.4, # couplage C ← D
            'eta': 0.2,     # production de dissipation par C²
            'zeta': 0.5,    # rappel dissipation
            'mu': 0.1       # couplage non linéaire supplémentaire (optionnel)
        }
        if params:
            self.params.update(params)

    def flow_field(self, phi, t):
        """
        Équations canoniques TTU-MC³ modifiées avec cible.
        """
        M, C, D = phi
        Mt, Ct, Dt = self.target
        p = self.params

        dM = -p['alpha'] * (M - Mt) + p['beta'] * C
        dC = -p['gamma'] * (C - Ct) + p['delta'] * M - p['epsilon'] * D
        dD = -p['zeta'] * (D - Dt) + p['eta'] * C**2 + p['mu'] * M * C

        return [dM, dC, dD]

    def solve(self, t_max=30, n_points=600):
        t = np.linspace(0, t_max, n_points)
        phi0 = [0.0, 0.0, 0.0]   # état initial neutre
        traj = odeint(self.flow_field, phi0, t)
        return t, traj

    def compute_convergence_metrics(self, traj):
        """Calcule le temps de stabilisation et l'erreur finale."""
        M, C, D = traj[-1]
        Mt, Ct, Dt = self.target
        error = np.sqrt((M-Mt)**2 + (C-Ct)**2 + (D-Dt)**2)
        # Trouver l'instant où le système entre dans une boule de rayon 0.05 autour de la cible
        diff = np.linalg.norm(traj - self.target, axis=1)
        t_stab = None
        for i, d in enumerate(diff):
            if d < 0.05:
                t_stab = i * (traj.shape[0] / diff.size)   # approximatif
                break
        return error, t_stab


# ============================================================================
# MODULE 3 : INTERFACE STREAMLIT
# ============================================================================
st.set_page_config(page_title="VTM Core - TTU-MC³", layout="wide")
st.title("🌐 Virtual Triadic Machine (VTM)")
st.markdown("**Stabilisation informationnelle par attracteur triadique**")
st.caption("Basé sur la Théorie Triadique Unifiée - Modèle de Cohérence Cubique (TTU-MC³)")

# --- Sidebar pour paramètres dynamiques ---
st.sidebar.header("⚙️ Paramètres du flot TTU")
params = {}
params['alpha'] = st.sidebar.slider("α (rappel mémoire)", 0.1, 1.5, 0.6, 0.05)
params['beta']  = st.sidebar.slider("β (couplage M ← C)", 0.0, 1.0, 0.2, 0.05)
params['gamma'] = st.sidebar.slider("γ (rappel cohérence)", 0.1, 1.5, 0.5, 0.05)
params['delta'] = st.sidebar.slider("δ (couplage C ← M)", 0.0, 1.0, 0.3, 0.05)
params['epsilon'] = st.sidebar.slider("ε (couplage C ← D)", 0.0, 1.0, 0.4, 0.05)
params['eta']    = st.sidebar.slider("η (production dissipation)", 0.0, 1.0, 0.2, 0.05)
params['zeta']   = st.sidebar.slider("ζ (rappel dissipation)", 0.1, 1.5, 0.5, 0.05)
params['mu']     = st.sidebar.slider("μ (non-linéarité)", 0.0, 1.0, 0.1, 0.05)

# --- Upload fichier ---
uploaded_file = st.file_uploader("Injecter un document (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    with st.spinner("Analyse du champ informationnel en cours..."):
        processor = TTUProcessor()
        text = processor.extract_text(uploaded_file)

        if len(text.strip()) == 0:
            st.error("Impossible d'extraire du texte de ce fichier.")
        else:
            signature = processor.compute_signature(text)
            st.info(f"Attracteur cible : M = {signature[0]:.4f}, C = {signature[1]:.4f}, D = {signature[2]:.4f}")

            # Simulation
            vtm = VTM_Core(signature, params)
            t, traj = vtm.solve(t_max=40, n_points=800)

            # Métriques
            error, t_stab = vtm.compute_convergence_metrics(traj)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mémoire finale (M)", f"{traj[-1,0]:.4f}")
            col2.metric("Cohérence finale (C)", f"{traj[-1,1]:.4f}")
            col3.metric("Dissipation finale (D)", f"{traj[-1,2]:.4f}")
            col4.metric("Erreur d'atteinte", f"{error:.1e}")

            if t_stab is not None:
                st.success(f"Stabilisation atteinte en t ≈ {t_stab:.2f} s (échelle simulée)")

            # --- Graphiques ---
            st.subheader("📈 Dynamique de stabilisation")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.plot(t, traj[:,0], label="Mémoire (M)", color='blue')
            ax1.plot(t, traj[:,1], label="Cohérence (C)", color='green')
            ax1.plot(t, traj[:,2], label="Dissipation (D)", color='red')
            ax1.axhline(y=signature[0], color='blue', linestyle='--', alpha=0.5)
            ax1.axhline(y=signature[1], color='green', linestyle='--', alpha=0.5)
            ax1.axhline(y=signature[2], color='red', linestyle='--', alpha=0.5)
            ax1.set_xlabel("Temps (unités arbitraires)")
            ax1.set_ylabel("Amplitude")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Évolution temporelle")

            ax2.plot(traj[:,0], traj[:,1], color='purple', alpha=0.7)
            ax2.scatter(signature[0], signature[1], color='red', s=100, zorder=5, label="Attracteur cible")
            ax2.set_xlabel("Mémoire")
            ax2.set_ylabel("Cohérence")
            ax2.set_title("Portrait de phase (M vs C)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            # --- Génération PDF ---
            def generate_pdf_report(traj, target, params, error):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4

                # Titre
                c.setFont("Helvetica-Bold", 16)
                c.drawString(2*cm, height - 2*cm, "Rapport d'analyse VTM - TTU-MC³")
                c.setFont("Helvetica", 10)
                c.drawString(2*cm, height - 2.5*cm, f"Date de génération : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

                # Signature cible
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2*cm, height - 4*cm, "Attracteur cible :")
                c.setFont("Helvetica", 10)
                c.drawString(2*cm, height - 4.5*cm, f"Mémoire (M) : {target[0]:.4f}")
                c.drawString(2*cm, height - 5.0*cm, f"Cohérence (C) : {target[1]:.4f}")
                c.drawString(2*cm, height - 5.5*cm, f"Dissipation (D) : {target[2]:.4f}")

                # État final
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2*cm, height - 7*cm, "État final :")
                c.setFont("Helvetica", 10)
                c.drawString(2*cm, height - 7.5*cm, f"M = {traj[-1,0]:.4f}")
                c.drawString(2*cm, height - 8.0*cm, f"C = {traj[-1,1]:.4f}")
                c.drawString(2*cm, height - 8.5*cm, f"D = {traj[-1,2]:.4f}")
                c.drawString(2*cm, height - 9.0*cm, f"Erreur résiduelle : {error:.2e}")

                # Paramètres
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2*cm, height - 10*cm, "Paramètres du flot TTU :")
                c.setFont("Helvetica", 8)
                y = height - 10.5*cm
                for key, val in params.items():
                    c.drawString(2*cm, y, f"{key} = {val:.3f}")
                    y -= 0.3*cm

                # Sauvegarde de la figure en image temporaire
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
                    img = ImageReader(tmp.name)
                    c.drawImage(img, 2*cm, 5*cm, width=12*cm, height=8*cm, preserveAspectRatio=True)

                c.showPage()
                c.save()
                return buffer.getvalue()

            pdf_data = generate_pdf_report(traj, signature, params, error)
            st.download_button(
                label="📄 Télécharger le rapport PDF",
                data=pdf_data,
                file_name="vtm_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("📄 Veuillez charger un document (PDF, DOCX, TXT) pour initialiser le calcul.")