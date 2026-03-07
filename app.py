import streamlit as st
from oracle_engine_v14 import OracleEngine
from ttu_file_scanner import scan_file
from analytics_engine import *
from github_memory import backup_memory

oracle=OracleEngine()

st.title("ORACLE V14 Ω TTU")

st.metric("Memories",oracle.stats())

text=st.text_area("Dialogue avec ORACLE")

if st.button("Envoyer"):

    oracle.learn(text)

    r=oracle.reason(text)

    st.write(r)

uploaded=st.file_uploader("Nourriture cérébrale",type=["txt","pdf","docx","csv","json"])

if uploaded:

    content=scan_file(uploaded)

    oracle.learn(content,"document")

    st.success("Document analysé et appris")

if st.button("Exporter rapport"):

    df=export_csv("oracle_memory/oracle.db")

    export_json(df)

    export_txt(df)

    st.success("Rapports générés")

if st.button("Graph évolution"):

    img=graph_progress("oracle_memory/oracle.db")

    st.image(img)

if st.button("Backup GitHub"):

    token=st.secrets["GITHUB_TOKEN"]

    repo=st.secrets["GITHUB_REPO"]

    backup_memory(token,repo,"oracle_memory")

    st.success("Backup GitHub effectué")