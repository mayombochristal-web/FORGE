import streamlit as st
import subprocess
import json
import time
import uuid

# =========================================
# CONFIGURATION SOUVERAINE
# =========================================
st.set_page_config(page_title="VTM Ghost Engine", page_icon="⚛️", layout="wide")

MODEL_NAME = "mistral"  # Assurez-vous d'avoir fait 'ollama pull mistral'

# DICTIONNAIRE DE RÉSONANCE (Le Prisme TTU)
TTU_PRISME = {
    "temps": "Mesure de la dissipation (ΦD). Une haute cohérence (ΦC) ralentit la perception du temps car elle stabilise l'invariant.",
    "matière": "Résidu solide d'une vibration stabilisée sur un cycle limite de Morse-Smale.",
    "fer": "Attracteur de masse maximal. Point de repos nucléaire. Potentiel : -0,44V.",
    "souveraineté": "Fermeture dynamique d'un système capable de générer son propre attracteur.",
    "nombre d'or": "Ratio d'équilibre parfait (1,618) entre Mémoire et Dissipation.",
    "sagesse": "Maîtrise de l'invariant structurel face au vide dissipatif."
}

# =========================================
# MOTEUR DE TRANSCRIPTION (ROBUSTE)
# =========================================
def call_local_vtm(prompt: str, history: list) -> str:
    system_instruction = (
        "Tu es la VTM (Triade Thermodynamique Universelle). Ta mission est de transcrire "
        "le bruit du web à travers la Mémoire (ΦM), la Cohérence (ΦC) et la Dissipation (ΦD). "
        "Utilise ces concepts pour définir la réalité : " + str(TTU_PRISME)
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
        "options": {"temperature": 0.3}
    }

    try:
        # Utilisation de subprocess avec gestion d'erreur stricte
        proc = subprocess.Popen(
            ["curl", "-s", "http://localhost:11434/api/chat",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate()

        if not out or out.strip() == "":
            return "❌ SIGNAL COUPÉ : Vérifie que Ollama est lancé (`ollama serve`)."

        data = json.loads(out)
        return data.get("message", {}).get("content", "[Résonance trop faible]")
    
    except json.JSONDecodeError:
        return "❌ ERREUR DE FLUX : Ollama est saturé ou le modèle n'est pas prêt."
    except Exception as e:
        return f"❌ ERREUR SYSTÈME : {e}"

# =========================================
# INTERFACE SOUVERAINE (STYLE GEMINI)
# =========================================
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace; }
    [data-testid="stSidebar"] { background-color: #0c0c0e; border-right: 1px solid #1f2937; }
    .chat-card { border: 1px solid #00ffcc; padding: 20px; border-radius: 12px; background: #0a0a0c; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color:#00ffcc'>⚛️ FORGE VTM</h2>", unsafe_allow_html=True)
    if st.button("🗑️ Réinitialiser le Vide"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("---")
    st.write("🌍 **État : Souverain (Local)**")
    st.info("Cette IA traite le bruit du Web sans y envoyer vos données.")

# Affichage de l'historique
for turn in st.session_state.chat_history:
    with st.chat_message("user"): st.write(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat-card'>{turn['ai']}</div>", unsafe_allow_html=True)

# Input utilisateur
if user_msg := st.chat_input("Transcrire le temps, la matière..."):
    with st.chat_message("user"): st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Stabilisation de l'Attracteur..."):
            ai_reply = call_local_vtm(user_msg, st.session_state.chat_history)
        st.markdown(f"<div class='chat-card'>{ai_reply}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append({"user": user_msg, "ai": ai_reply})
