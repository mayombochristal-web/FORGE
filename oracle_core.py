# =====================================================
# 🧠 ORACLE CORE V6.1 — AUTO PERSISTENT CORTEX
# Cerveau vivant (indépendant UI)
# =====================================================

import os
import json
import base64
import time
import requests
import streamlit as st

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")

os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================
# SAFE SESSION INIT
# =====================================================

def init_session():

    defaults = {
        "memory_dirty": False,
        "last_change": 0,
        "last_sync_check": 0
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# =====================================================
# MEMORY LOAD
# =====================================================

def load_memory():

    if not os.path.exists(MEMORY_FILE):

        memory = {"messages": []}

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

        return memory

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)

        # sécurité structure
        if "messages" not in memory:
            memory["messages"] = []

        return memory

    except Exception:
        return {"messages": []}


# =====================================================
# MEMORY SAVE LOCAL
# =====================================================

def save_memory_local(memory):

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# =====================================================
# DIRTY FLAG (activité neuronale)
# =====================================================

def mark_dirty():

    st.session_state["memory_dirty"] = True
    st.session_state["last_change"] = time.time()


# =====================================================
# ADD MESSAGE
# =====================================================

def add_message(role, content):

    memory = load_memory()

    memory["messages"].append({
        "role": role,
        "content": content,
        "time": time.time()
    })

    # limite mémoire anti JSON wall
    if len(memory["messages"]) > 2000:
        memory["messages"] = memory["messages"][-1500:]

    save_memory_local(memory)
    mark_dirty()


# =====================================================
# GITHUB SYNC ENGINE
# =====================================================

def github_sync():

    # rien à sync
    if not st.session_state.get("memory_dirty"):
        return

    # secrets non configurés → skip silencieux
    if "GITHUB_TOKEN" not in st.secrets:
        return
    if "GITHUB_REPO" not in st.secrets:
        return

    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{repo}/contents/{MEMORY_FILE}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    try:

        with open(MEMORY_FILE, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        # récupérer SHA si fichier existe
        r = requests.get(url, headers=headers, timeout=10)

        sha = None
        if r.status_code == 200:
            sha = r.json().get("sha")

        payload = {
            "message": "🧠 Oracle auto memory sync",
            "content": encoded,
            "branch": branch
        }

        if sha:
            payload["sha"] = sha

        requests.put(url, headers=headers, json=payload, timeout=15)

        st.session_state["memory_dirty"] = False

    except Exception:
        # jamais casser l'app
        pass


# =====================================================
# AUTO SYNC LOOP (NON BLOQUANT)
# =====================================================

def auto_sync_loop(delay=20):
    """
    Synchronisation périodique légère.
    Appelée à chaque rerun Streamlit.
    """

    now = time.time()
    last = st.session_state.get("last_sync_check", 0)

    if now - last > delay:
        github_sync()
        st.session_state["last_sync_check"] = now