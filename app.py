import streamlit as st
from oracle_core import process_input, brain

st.set_page_config(page_title="ORACLE V3.2",page_icon="🧠")

st.title("🧠 ORACLE V3.2 — Agent Cognitif")

msg = st.text_input("Parlez à l'Oracle")

file = st.file_uploader(
    "Insérer document / audio",
    type=["pdf","docx","txt","csv","wav"]
)

if st.button("Envoyer"):
    reply = process_input(msg,file)
    st.write(reply)

for m in brain["dialog_memory"]:
    st.write(m)

with st.sidebar:

    st.header("🧠 État Cognitif")

    for k,v in brain["phi"].items():
        st.progress(v,text=f"{k}: {v:.2f}")

    st.write("Identity entropy:",
             round(brain["identity_entropy"],3))