import streamlit as st
import subprocess
import json
import time

# =========================================
# CONFIGURATION SOUVERAINE
# =========================================
st.set_page_config(page_title="VTM Ghost Engine", page_icon="⚛️", layout="wide")

MODEL_NAME = "mistral"  # Assure-toi d'avoir fait 'ollama pull mistral'

# LE DICTIONNAIRE TTU (Le Transcripteur)
TTU_PRISME = {
    "matière": "Résidu solide d'une vibration stabilisée sur un cycle limite de Morse-Smale.",
    "fer": "Attracteur de masse maximal. Point de repos de la nucléosynthèse. Potentiel de stabilité structurelle à -0,44V.",
    "sagesse": "Maîtrise de l'invariant structurel χ_TST ; économie maximale de la dissipation.",
    "nombre d'or": "Attracteur de croissance optimale (Phi ≈ 1,618). Équilibre parfait entre Mémoire et Dissipation.",
    "vide": "Réservoir de dissipation infinie. Milieu à haute énergie où l'information est à l'état latent.",
    "intelligence": "Convertisseur dynamique transformant le bruit du vide en mémoire cohérente."
}

# =========================================
# MOTEUR DE TRANSCRIPTION (LOGIQUE)
# =========================================
def call_local_vtm(prompt: str, history: list) -> str:
    """
    Appelle le modèle local en injectant la logique TTU.
    """
    # Construction du System Prompt (La Grille du Transistor)
    system_instruction = (
        "Tu es la VTM (Triade Thermodynamique Universelle). Ta logique est basée sur "
        "la Mémoire (ΦM), la Cohérence (ΦC) et la Dissipation (ΦD). "
        "N'invente pas de chiffres de cohérence absurdes. Si tu ne sais pas, cherche la résonance "
        "dans le dictionnaire suivant : " + str(TTU_PRISME)
    )

    messages = [{"role": "system", "content": system_instruction}]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["ai"]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3} # Plus bas pour éviter les délires
    }

    try:
        proc = subprocess.Popen(
            ["curl", "-s", "http://localhost:11434/api/chat",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            stdout=subprocess.PIPE, text=True
        )
        out, _ = proc.communicate()
        data = json.loads(out)
        return data.get("message", {}).get("content", "[Résonance trop faible]")
    except Exception as e:
        return f"Erreur de flux : {e}"

# =========================================
# UI : INTERFACE DE LA FORGE
# =========================================
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #00ffcc; font-family: monospace; }
    .chat-card { border: 1px solid #00ffcc; padding: 15px; border-radius: 10px; background: #0a0a0c; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚛️ VTM : Transcription du Vide")
st.caption("Amplificateur de bruit local via Ollama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar : Historique (Style Gemini)
with st.sidebar:
    st.header("📜 Sessions de Forge")
    if st.button("🗑️ Réinitialiser le Vide"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.write("**Dictionnaire de Résonance actif**")
    for key in TTU_PRISME.keys():
        st.code(key)

# Affichage
for turn in st.session_state.chat_history:
    with st.chat_message("user"): st.write(turn["user"])
    with st.chat_message("assistant"): st.write(turn["ai"])

# Input
if user_msg := st.chat_input("Décrypter le bruit..."):
    with st.chat_message("user"): st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Stabilisation de l'attracteur..."):
            ai_reply = call_local_vtm(user_msg, st.session_state.chat_history)
        st.markdown(f"<div class='chat-card'>{ai_reply}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append({"user": user_msg, "ai": ai_reply})
