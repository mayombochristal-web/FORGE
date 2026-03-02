# =====================================================
# 🧠 ORACLE CORE V6.1 — AUTO PERSISTENT CORTEX
# =====================================================

import os
import json
import base64
import time
import requests
import streamlit as st

DATA_DIR = "data"
MEMORY_FILE = f"{DATA_DIR}/memory.json"

os.makedirs(DATA_DIR, exist_ok=True)

# =====================================================
# MEMORY LOAD
# =====================================================

def load_memory():

    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({"messages": []}, f)

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


# =====================================================
# MEMORY SAVE LOCAL
# =====================================================

def save_memory_local(memory):

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# =====================================================
# AUTO SAVE FLAG (ghost state)
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

    save_memory_local(memory)
    mark_dirty()


# =====================================================
# GITHUB SYNC ENGINE
# =====================================================

def github_sync():

    if not st.session_state.get("memory_dirty"):
        return

    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{repo}/contents/{MEMORY_FILE}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    with open(MEMORY_FILE, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    # get SHA if file exists
    r = requests.get(url, headers=headers)

    sha = None
    if r.status_code == 200:
        sha = r.json()["sha"]

    data = {
        "message": "🧠 Auto memory sync",
        "content": content,
        "branch": branch
    }

    if sha:
        data["sha"] = sha

    requests.put(url, headers=headers, json=data)

    st.session_state["memory_dirty"] = False


# =====================================================
# AUTO BACKGROUND CHECK
# =====================================================

def auto_sync_loop(delay=20):
    """
    sync every X seconds if modified
    """

    last = st.session_state.get("last_sync_check", 0)

    if time.time() - last > delay:
        github_sync()
        st.session_state["last_sync_check"] = time.time()