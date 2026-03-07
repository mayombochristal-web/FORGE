# ============================================================
# ORACLE V14 Ω STREAMLIT INTERFACE
# ============================================================

import streamlit as st
from oracle_engine_v14 import OracleEngine
from github import Github
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="ORACLE V14 Ω",layout="wide")

oracle=OracleEngine()

# ============================================================
# GITHUB CONFIG
# ============================================================

try:

    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
    GITHUB_MEMORY_DIR = st.secrets["GITHUB_MEMORY_DIR"]
    GITHUB_BRANCH = st.secrets["GITHUB_BRANCH"]

    github=Github(GITHUB_TOKEN)
    repo=github.get_repo(GITHUB_REPO)

except:

    github=None

# ============================================================
# HEADER
# ============================================================

st.title("🧠 ORACLE V14 Ω")

col1,col2=st.columns(2)

with col1:
    st.metric("Memories",oracle.memory_size())

with col2:
    st.metric("Status","ACTIVE")

# ============================================================
# INPUT
# ============================================================

user_input=st.text_area("Dialogue avec ORACLE")

if st.button("Envoyer"):

    if user_input:

        oracle.store(user_input)

        response=oracle.reason(user_input)

        st.success(response)

# ============================================================
# GITHUB SAVE
# ============================================================

if st.button("Sauvegarde mémoire GitHub"):

    if github:

        for root,dirs,files in os.walk("oracle_memory"):

            for f in files:

                path=os.path.join(root,f)

                with open(path,"rb") as file:

                    content=file.read()

                repo.create_file(
                    f"{GITHUB_MEMORY_DIR}/{f}",
                    "memory backup",
                    content,
                    branch=GITHUB_BRANCH
                )

        st.success("Mémoire sauvegardée sur GitHub")

    else:

        st.warning("GitHub non configuré")