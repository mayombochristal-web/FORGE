# =====================================================
# 🧠 ORACLE V6 — INTERFACE STREAMLIT
# Utilise le cœur cognitif unifié
# =====================================================

import streamlit as st
import pandas as pd
import PyPDF2
import docx
import speech_recognition as sr
import json
import os
import time

from oracle_core import OracleBrain

# ---------- Initialisation ----------
if "brain" not in st.session_state:
    st.session_state.brain = OracleBrain("oracle_memory.json")

brain = st.session_state.brain

# ---------- Fonctions d'extraction multimodale ----------
def extract_from_file(uploaded_file):
    """Extrait le texte d'un fichier uploadé."""
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join(p.extract_text() or "" for p in reader.pages)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return " ".join(p.text for p in doc.paragraphs)
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        elif uploaded_file.type == "audio/wav":
            r = sr.Recognizer()
            with sr.AudioFile(uploaded_file) as source:
                audio = r.record(source)
            return r.recognize_google(audio, language="fr-FR")
        else:
            return ""
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        return ""

# ---------- Configuration page ----------
st.set_page_config(page_title="ORACLE V6", page_icon="🧠", layout="wide")
st.title("🧠 ORACLE V6 — Cognition Unifiée")

# ---------- Onglets ----------
tab1, tab2 = st.tabs(["💬 Conversation", "📚 Nourrir (fichiers)"])

# ========== ONGLET CONVERSATION ==========
with tab1:
    st.subheader("Parlez à l'Oracle")

    # Affichage de la conversation
    for msg in brain.dialog_memory:
        st.write(msg)

    # Zone de saisie
    user_msg = st.text_input("Votre message", key="conv_input")
    col1, col2 = st.columns([1,5])
    with col1:
        send = st.button("Envoyer")

    if send and user_msg:
        with st.spinner("L'Oracle réfléchit..."):
            response = brain.process_input(user_msg, is_user=True)
        # L'affichage se met à jour via st.rerun implicite ? 
        # On force le rerun pour voir le nouveau message
        st.rerun()

# ========== ONGLET NOURRIR ==========
with tab2:
    st.subheader("Nourrir l'Oracle avec des fichiers")

    mode = st.radio("Type de source", ["Texte", "Document", "Excel", "Audio"])
    content = ""

    if mode == "Texte":
        content = st.text_area("Collez votre texte ici")
    else:
        file = st.file_uploader("Choisissez un fichier", 
                                type=["pdf","docx","txt","csv","wav"] if mode!="Excel" else ["xlsx","xls"])
        if file:
            content = extract_from_file(file)

    if st.button("🧠 Nourrir et générer"):
        if content:
            with st.spinner("L'Oracle assimile et répond..."):
                # Le contenu est traité comme un message utilisateur, ce qui déclenche tout le pipeline
                response = brain.process_input(content, is_user=True)
            st.success("Contenu appris. Réponse générée :")
            st.write(response)
            st.rerun()
        else:
            st.warning("Aucun contenu à apprendre.")

# ---------- Sidebar : état cognitif ----------
with st.sidebar:
    st.header("🧠 État cognitif")

    # Taille de la mémoire
    mem_size = os.path.getsize(brain.memory_file) / 1024 if os.path.exists(brain.memory_file) else 0
    st.metric("Mémoire (Ko)", f"{mem_size:.2f}")

    st.divider()
    st.subheader("Φ Dynamique")
    for k, v in brain.phi.items():
        st.progress(v, text=f"{k} : {v:.2f}")

    st.divider()
    st.subheader("👻 Fantôme")
    st.progress(brain.ghost_activity, text="Influence")
    if st.button("Voir mémoire fantôme"):
        st.write(brain.ghost_memory)

    st.divider()
    st.subheader("🌙 Sommeil")
    last = time.strftime("%H:%M:%S", time.localtime(brain.last_sleep))
    st.caption(f"Dernier sommeil : {last}")
    if st.button("Forcer le sommeil"):
        brain.sleep_cycle()
        st.success("Sommeil effectué")
        time.sleep(1)
        st.rerun()

    st.divider()
    st.subheader("💾 Sauvegarde")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Télécharger mémoire",
            data=json.dumps(brain.lexicon, indent=2, ensure_ascii=False),
            file_name="oracle_memory.json"
        )
    with col_b:
        restore = st.file_uploader("Restaurer", type="json", key="restore")
        if restore:
            try:
                new_lex = json.load(restore)
                brain.lexicon = new_lex
                brain._save_lex()
                st.success("Mémoire restaurée")
                st.rerun()
            except:
                st.error("Fichier invalide")