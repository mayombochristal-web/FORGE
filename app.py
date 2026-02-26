# ==========================================================
# 🔥 ORACLE V13 — CHATBOT IA MODERNE AUTONOME
# Production Ready — Streamlit + Ollama
# ==========================================================

import streamlit as st
import numpy as np
import json
import os
import time
import requests
from typing import List, Dict

# ==========================================================
# CONFIG
# ==========================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED = "http://localhost:11434/api/embeddings"

MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

MEMORY_FILE = "oracle_memory.json"

# ==========================================================
# UTILITIES
# ==========================================================

def call_llm(messages, temperature=0.7, max_tokens=800):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"Erreur LLM: {e}"


def get_embedding(text):
    try:
        r = requests.post(
            OLLAMA_EMBED,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        return np.array(r.json()["embedding"])
    except:
        return np.random.randn(768)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ==========================================================
# MEMORY SYSTEM
# ==========================================================

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))


def save_memory(prompt, answer):
    memory = load_memory()
    emb = get_embedding(prompt).tolist()

    memory.append({
        "prompt": prompt,
        "answer": answer,
        "embedding": emb
    })

    memory = memory[-200:]  # limite mémoire
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"))


def retrieve_memories(prompt, k=3):
    memory = load_memory()
    if not memory:
        return ""

    emb = get_embedding(prompt)

    scored = []
    for m in memory:
        score = cosine(emb, np.array(m["embedding"]))
        scored.append((score, m))

    scored.sort(reverse=True)

    return "\n".join(
        f"- {x[1]['prompt']}" for x in scored[:k]
    )


# ==========================================================
# LATENT SPACE
# ==========================================================

def init_latent():
    return {
        "profondeur": 1.2,
        "coherence": 1.5,
        "exploration": 1.0,
        "rigueur": 1.2,
    }


def latent_to_params(latent):
    temp = 0.7 * latent["exploration"] / latent["rigueur"]
    temp = float(np.clip(temp, 0.2, 1.2))

    max_tokens = int(600 * latent["profondeur"])

    return temp, max_tokens


# ==========================================================
# PLANNER
# ==========================================================

def build_plan(prompt):
    messages = [
        {"role": "system",
         "content": "Planifie brièvement la meilleure réponse."},
        {"role": "user", "content": prompt},
    ]

    return call_llm(messages, temperature=0.3, max_tokens=200)


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

def build_system_prompt(latent):

    return f"""
Tu es ORACLE, un assistant IA moderne.

Objectifs :
- naturel et clair
- utile et intelligent
- conversation fluide
- profondeur si nécessaire
- concision sinon

Ne parle jamais de ton fonctionnement interne.

Paramètres internes :
{json.dumps(latent, ensure_ascii=False)}
"""


# ==========================================================
# GENERATION PIPELINE
# ==========================================================

def generate_answer(prompt, history, latent):

    memories = retrieve_memories(prompt)
    plan = build_plan(prompt)

    temp, max_tokens = latent_to_params(latent)

    condensed_history = ""
    for h in history[-6:]:
        condensed_history += f"{h['role']}:{h['content']}\n"

    messages = [
        {"role": "system", "content": build_system_prompt(latent)},
        {
            "role": "user",
            "content": f"""
PLAN INTERNE:
{plan}

Souvenirs pertinents:
{memories}

Conversation:
{condensed_history}

Message utilisateur:
{prompt}
"""
        },
    ]

    first = call_llm(messages, temp, max_tokens)

    # ===== AUTO REFINE =====

    refine_messages = [
        {
            "role": "system",
            "content": "Améliore la réponse pour clarté et cohérence."
        },
        {
            "role": "user",
            "content": f"Question:{prompt}\nRéponse:{first}"
        },
    ]

    refined = call_llm(refine_messages, temp - 0.1, max_tokens)

    return refined


# ==========================================================
# INTERNAL QUALITY SCORE
# ==========================================================

def internal_score(answer):
    r = call_llm([
        {"role": "system", "content": "Note la qualité de 1 à 5."},
        {"role": "user", "content": answer},
    ], temperature=0.2, max_tokens=5)

    try:
        return float(r.strip()[0])
    except:
        return 3.0


def update_latent(latent, user_rating, internal):

    reward = (user_rating - 3)*0.6 + (internal - 3)*0.4

    latent["profondeur"] = float(np.clip(
        latent["profondeur"] + reward*0.05, 0.8, 2.5))

    latent["exploration"] = float(np.clip(
        latent["exploration"] + reward*0.03, 0.7, 1.5))

    latent["rigueur"] = float(np.clip(
        latent["rigueur"] + reward*0.02, 0.8, 1.6))

    return latent


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(page_title="🔥 Oracle V13", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

if "latent" not in st.session_state:
    st.session_state.latent = init_latent()

# Sidebar
with st.sidebar:
    st.title("🔥 Oracle V13")
    st.caption("Chatbot IA moderne local")

    if st.button("Reset mémoire"):
        st.session_state.history = []
        st.session_state.latent = init_latent()

    st.divider()
    st.json(st.session_state.latent)

# Display chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Parlez avec Oracle...")

if prompt:

    st.session_state.history.append(
        {"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        answer = generate_answer(
            prompt,
            st.session_state.history,
            st.session_state.latent
        )

        # pseudo streaming
        out = ""
        for w in answer.split():
            out += w + " "
            placeholder.markdown(out)
            time.sleep(0.01)

    st.session_state.history.append(
        {"role": "assistant", "content": answer})

    save_memory(prompt, answer)

    # ===== FEEDBACK =====
    rating = st.slider("Qualité de la réponse", 1, 5, 4)

    if st.button("Valider apprentissage"):
        internal = internal_score(answer)

        st.session_state.latent = update_latent(
            st.session_state.latent,
            rating,
            internal
        )

        st.success("Oracle a appris de cette interaction.")