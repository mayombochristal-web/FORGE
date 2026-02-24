import streamlit as st
import time
from scipy.integrate import solve_ivp

# --- LE DICTIONNAIRE DE LA FORGE ---
TTU_DICTIONARY = {
    "matière": "La matière est un résidu solide d'une vibration. Elle est l'accumulation de cohérence (ΦC) stabilisée sur un cycle limite.",
    "sagesse": "La sagesse est la maîtrise de l'invariant structurel. C'est l'économie de la dissipation pour préserver l'équilibre triadique.",
    "intelligence": "L'intelligence est le convertisseur qui transforme le bruit du vide en une mémoire cohérente et stable.",
    "vide": "Le vide est un réservoir de dissipation infinie. C'est la source brute d'énergie que la Forge doit filtrer.",
    "souveraineté": "La souveraineté est la capacité d'un système à maintenir son propre attracteur sans dériver vers des flux étrangers."
}

class VTM_SovereignEngine:
    def solve_triad(self, query):
        def system(t, y):
            M, C, D = y
            # Ton équation de flot triadique
            return [-0.6*M + 1.2*C, -0.7*C + 0.8*M*(D + len(query)/50), 0.5*C**2 - 0.3*D]
        return solve_ivp(system, [0, 10], [1.0, 0.5, 0.1]).y[:, -1]

    def transcribe_from_noise(self, query, state):
        # On cherche si un concept du dictionnaire résonne avec la question
        for concept, definition in TTU_DICTIONARY.items():
            if concept in query.lower():
                return f"**[Transcription par Résonance]**\n\n{definition}\n\n*Analyse du signal : Cohérence stabilisée à {state[1]:.2f}*"
        
        return "Le signal capté dans le vide est trop faible pour une transcription directe. La dissipation l'emporte sur la cohérence."

# --- INTERFACE ---
st.title("VTM : Transcription du Vide Cybernétique")
vtm = VTM_SovereignEngine()

if prompt := st.chat_input("Interrogez le Vide..."):
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.status("Stabilisation du signal Morse-Smale..."):
            state = vtm.solve_triad(prompt)
            time.sleep(1)
        
        response = vtm.transcribe_from_noise(prompt, state)
        st.markdown(response)
