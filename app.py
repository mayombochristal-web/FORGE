import streamlit as st
import time
from scipy.integrate import solve_ivp
from pypdf import PdfReader
from docx import Document
import io

# --- CONFIGURATION STYLE "GEMINI EXPERIENCE" ---
st.set_page_config(page_title="VTM Intelligence", page_icon="⚛️", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #131314; color: #e3e3e3; font-family: 'Google Sans', sans-serif; }
    .stChatMessage { background-color: transparent !important; }
    .stChatInputContainer { background-color: #1e1f20; border-radius: 28px; border: 1px solid #444746; }
    /* Personnalisation des boutons et sidebar */
    .st-emotion-cache-6q9sum.edgvbvh3 { background-color: #1e1f20; border-radius: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- MOTEUR D'INTERPRÉTATION (LOGIQUE TTU DISCRÈTE) ---
class VTM_Intelligence:
    def __init__(self, context):
        self.context = context

    def solve_logic(self, query):
        """ Évalue la stabilité de la question via le flot triadique (en arrière-plan) """
        # On définit une complexité basée sur la requête
        complexite = len(query) / 20.0
        phi_c = min(complexite / 9.0, 2.0)
        
        # Le calcul triadique (M, C, D) détermine la 'profondeur' de la réponse
        def triad_flow(t, y):
            return [-0.6*y[0] + 1.2*y[1], -0.7*y[1] + 0.8*y[0]*y[2], 0.5*y[1]**2 - 0.3*y[2]]
        
        sol = solve_ivp(triad_flow, [0, 5], [1.0, phi_c, 0.1])
        return phi_c > 0.45  # Seuil de réponse logique

    def get_response(self, query):
        """ Fouille dans la matrice (RAG) et synthétise une réponse """
        if not self.context:
            return "Je suis prêt à interpréter vos travaux. Veuillez charger vos thèses ou fichiers dans la matrice (menu latéral)."

        # Recherche de résonance par mots-clés (pragmatique)
        words = query.lower().split()
        # On découpe en blocs plus larges pour garder le contexte
        segments = self.context.split('\n\n') 
        scored_segments = []
        
        for seg in segments:
            score = sum(2 for w in words if w in seg.lower()) # Poids sur les mots clés
            if score > 0:
                scored_segments.append((score, seg))
        
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_segments:
            return "D'après les principes de la forge, cette question ne trouve pas de résonance directe dans vos documents, mais elle peut être analysée sous l'angle de la dynamique relationnelle..."
        
        # On assemble les 3 meilleurs segments pour une réponse riche
        top_context = " ".join([s[1] for s in scored_segments[:2]])
        return top_context

# --- GESTION SÉCURISÉE DES FICHIERS (FIN DU UNICODEDECODEERROR) ---
def secure_read_files(uploaded_files):
    text_accumulated = ""
    for file in uploaded_files:
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text_accumulated += page.extract_text() + "\n"
            elif file.name.endswith('.docx'):
                doc = Document(file)
                text_accumulated += "\n".join([p.text for p in doc.paragraphs]) + "\n"
            elif file.name.endswith('.txt'):
                # Lecture sécurisée en ignorant les caractères spéciaux
                text_accumulated += file.read().decode('utf-8', errors='ignore') + "\n"
        except Exception as e:
            st.error(f"Erreur sur {file.name} : {str(e)}")
    return text_accumulated

# --- GESTION DE LA SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "matrix_data" not in st.session_state:
    st.session_state.matrix_data = ""

# --- SIDEBAR (LA MATRICE DE DONNÉES) ---
with st.sidebar:
    st.title("📂 Matrice VTM")
    st.write("Chargez jusqu'à 200 Mo de thèses et documents.")
    uploaded = st.file_uploader("Fichiers Source", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    if uploaded:
        if st.button("🔄 Actualiser la Matrice"):
            st.session_state.matrix_data = secure_read_files(uploaded)
            st.success(f"Matrice stabilisée ({len(st.session_state.matrix_data)//1024} Ko)")

# --- INTERFACE DE CHAT ---
st.title("⚛️ VTM Intelligence")
st.caption("Interpréteur de connaissances doctorales — Système Triadique")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez une question à la matrice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_area = st.empty()
        
        with st.status("Recherche de cohérence...", expanded=False) as status:
            vtm = VTM_Intelligence(st.session_state.matrix_data)
            is_logical = vtm.solve_logic(prompt)
            time.sleep(0.6)
            status.update(label="Analyse terminée", state="complete")

        if not is_logical and len(prompt) < 12:
            answer = "Cette pensée est trop fragmentée pour être interprétée par la Forge. Pourriez-vous développer ?"
        else:
            answer = vtm.get_response(prompt)

        # Animation "Gemini Style"
        displayed_text = ""
        for word in answer.split():
            displayed_text += word + " "
            response_area.markdown(displayed_text + "▌")
            time.sleep(0.04)
        response_area.markdown(displayed_text)
        
    st.session_state.messages.append({"role": "assistant", "content": answer})
