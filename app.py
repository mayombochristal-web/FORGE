import streamlit as st
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document

# --- CONFIGURATION DE L'APPARENCE (STYLE GEMINI) ---
st.set_page_config(page_title="VTM Intelligence", page_icon="⚛️", layout="centered")

st.markdown("""
    <style>
    /* Style global sombre et épuré */
    .stApp { background-color: #131314; color: #e3e3e3; }
    .stChatMessage { background-color: transparent !important; border: none !important; }
    .stChatInputContainer { background-color: #1e1f20; border-radius: 30px; }
    /* Masquer le jargon technique par défaut */
    .stStatus { border: none; background: transparent; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR DE RÉFLEXION (BACKEND SILENCIEUX) ---
class ForgeEngine:
    def __init__(self, vault_text):
        self.vault_text = vault_text

    def think(self, query):
        """ Simule le processus de réflexion triadique sans afficher les équations """
        # Énergie de la question
        energy = len(query) / 10.0
        phi_c = energy / 9.0
        
        # Simulation de convergence (Calcul interne)
        def flow(t, y):
            return [-0.6*y[0] + 1.2*y[1], -0.7*y[1] + 0.8*y[0]*y[2], 0.5*y[1]**2 - 0.3*y[2]]
        
        sol = solve_ivp(flow, [0, 5], [1.0, phi_c, 0.1])
        is_stable = phi_c > 0.5088
        return is_stable

    def interpret(self, query):
        """ Recherche dans les 200 Mo de thèses la réponse la plus proche """
        if not self.vault_text:
            return "Veuillez injecter la matrice de données dans la barre latérale pour que je puisse consulter vos travaux."
        
        # Recherche par mots-clés dans tes documents
        words = query.lower().split()
        paragraphs = self.vault_text.split('.')
        results = []
        for p in paragraphs:
            score = sum(1 for w in words if w in p.lower())
            if score > 0:
                results.append((score, p.strip()))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        if not results:
            return "Je n'ai pas trouvé de résonance directe dans vos thèses pour cette question, mais selon la logique de la forge, voici ce qu'on peut en déduire..."
        
        return ". ".join([r[1] for r in results[:2]]) + "."

# --- INITIALISATION DE LA SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vault_content" not in st.session_state:
    st.session_state.vault_content = ""

# --- BARRE LATÉRALE (GESTION DES DOCUMENTS) ---
with st.sidebar:
    st.title("📂 Matrice")
    uploaded_files = st.file_uploader("Chargez vos thèses (PDF/DOCX)", accept_multiple_files=True)
    if uploaded_files:
        text = ""
        for f in uploaded_files:
            if f.type == "application/pdf":
                pdf = PdfReader(f); text += " ".join([p.extract_text() for p in pdf.pages])
            elif "document" in f.type:
                doc = Document(f); text += " ".join([p.text for p in doc.paragraphs])
            else:
                text += f.read().decode()
        st.session_state.vault_content = text
        st.success("Connaissances intégrées.")

# --- INTERFACE DE CHAT ---
st.title("⚛️ VTM Intelligence")

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez une question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'IA
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Étape de "Réflexion" (Simulation de l'IA qui cherche)
        with st.status("Réflexion...", expanded=False) as status:
            engine = ForgeEngine(st.session_state.vault_content)
            stable = engine.think(prompt)
            time.sleep(1) # Petit délai pour le feeling "IA"
            status.update(label="Analyse terminée", state="complete", expanded=False)

        # Génération de la réponse
        if not stable and len(prompt) < 10:
            final_answer = "Cette question semble trop fragmentée pour générer une réponse stable dans le cadre de vos thèses."
        else:
            final_answer = engine.interpret(prompt)

        # Effet de frappe progressive
        full_text = ""
        for chunk in final_answer.split():
            full_text += chunk + " "
            response_placeholder.markdown(full_text + "▌")
            time.sleep(0.04)
        response_placeholder.markdown(full_text)
        
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
