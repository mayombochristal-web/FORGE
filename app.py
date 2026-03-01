# =====================================================
# 🧠 ORACLE S+ — ARCHITECTURE COGNITIVE STABLE
# Pipeline officiel :
# S+01 → S+16
# =====================================================

# =====================================================
# S+01 — IMPORT_SYSTEM_CORE
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime, time
import xml.etree.ElementTree as ET
from collections import Counter
try:
    from spectral_module import spectral_ui
except ImportError:
    def spectral_ui(*args, **kwargs):
        pass
# =====================================================
# S+02 — STREAMLIT_PAGE_CONFIG
# =====================================================

st.set_page_config(page_title="ORACLE S+", layout="wide")

# =====================================================
# S+03 — MEMORY_PATH_MANAGER
# =====================================================

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
MEM = os.path.join(BASE_DIR, "oracle_memory")
os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

# =====================================================
# S+04 — SAFE_IO_LAYER
# =====================================================

def load_json(p):
    try:
        with open(p,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}
def save_json(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_frag():
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.to_csv(FILES["fragments"], index=False)

# =====================================================
# S+05 — MEMORY_INITIALIZER
# =====================================================

def init_memory():

    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(
            FILES["fragments"], index=False
        )

    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})

    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS":12,
            "age":0,
            "new_today":0,
            "last_day":str(datetime.date.today()),
            "timeline":[]
        })

init_memory()

# ==========================================================
# B2.5 — Cognitive Runtime Guard
# Stabilisation session + cohérence TTU runtime
# ==========================================================

# ----------------------------------------------------------
# Runtime ID unique (évite duplication session)
# ----------------------------------------------------------
def runtime_id():
    if "runtime_id" not in st.session_state:
        st.session_state.runtime_id = str(uuid.uuid4())
    return st.session_state.runtime_id


# ----------------------------------------------------------
# Horloge cognitive (∆t interne)
# ----------------------------------------------------------
def cognitive_clock():
    now = time.time()

    if "cognitive_time" not in st.session_state:
        st.session_state.cognitive_time = now
        st.session_state.delta_t = 0.0
    else:
        st.session_state.delta_t = now - st.session_state.cognitive_time
        st.session_state.cognitive_time = now

    return st.session_state.delta_t


# ----------------------------------------------------------
# Guard anti double-exécution Streamlit
# ----------------------------------------------------------
def runtime_guard():

    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.boot_time = time.time()
        st.session_state.rerun_count = 0
    else:
        st.session_state.rerun_count += 1


# ----------------------------------------------------------
# Synchronisation mémoire dialogue
# (corrige message invisible)
# ----------------------------------------------------------
def sync_dialog_memory():

    if "dialog" not in st.session_state:
        st.session_state.dialog = []

    if "pending_output" in st.session_state:
        st.session_state.dialog.append(
            st.session_state.pending_output
        )
        del st.session_state.pending_output


# ----------------------------------------------------------
# Bootstrap global
# ----------------------------------------------------------
def cognitive_bootstrap():

    runtime_guard()
    runtime_id()
    cognitive_clock()
    sync_dialog_memory()


# lancement automatique
cognitive_bootstrap()

# =====================================================
# S+06 — SHADOW_STATE_LOADER
# =====================================================

hash check
def sync_shadow():

    if "shadow_loaded" not in st.session_state:

        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])

        st.session_state.shadow_loaded = True

sync_shadow()

# =====================================================
# S+08 — TEXT_NORMALIZER_TOKENIZER
# =====================================================

def char_tokens(text):
    return [c for c in text.lower() if c.strip()]
    
def clean(t):
    return re.sub(r"[^\wàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

# =====================================================
# S+09 — LEARNING_ENGINE_CORE (OPTIMISÉ)
# =====================================================

def learn(text):
    chars = char_tokens(text)
    words = tokenize(text)
    if not words: return 0

    # 1. Mise à jour fragments (Méthode Dict pour la vitesse)
    df = st.session_state.shadow_frag
    counts = Counter(words)
    
    # Conversion en dictionnaire pour fusion rapide
    current_memory = dict(zip(df.fragment, df['count']))
    for w, c in counts.items():
        current_memory[w] = current_memory.get(w, 0) + c
    
    # Reconstruction du DataFrame
    new_df = pd.DataFrame(list(current_memory.items()), columns=["fragment", "count"])
    st.session_state.shadow_frag = new_df
    save_frag(new_df)

    # 2. Mise à jour Relations (TST)
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + 2
    save_json(FILES["relations"], assoc)

    # 3. Cortex (Physique Titan)
    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex["last_day"] != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today

    # Facteur de compression massive
    UNITE_MASSIVE = 250_000_000 
    cortex["age"] += len(chars) / UNITE_MASSIVE
    cortex["new_today"] += len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"] * 1000))

    save_json(FILES["cortex"], cortex)
    return len(words)
if df.empty:
    current_memory = {}
else:
    current_memory = dict(zip(df["fragment"], df["count"]))

# =====================================================
# S+10 — SEMANTIC_SEARCH_ENGINE
# =====================================================

def association_density():
    assoc=st.session_state.shadow_rel
    links=sum(len(v) for v in assoc.values())
    vocab=len(assoc)
    return round(links/max(vocab,1),2)

# =====================================================
# S+11 — PRETHINK_ENGINE
# =====================================================

def prethink(seed):

    assoc=st.session_state.shadow_rel

    if seed in assoc and assoc[seed]:
        return max(assoc[seed], key=assoc[seed].get)

    return seed
    
# =====================================================
# S+11B — LINGUISTIC OPERATOR L (TST)
# =====================================================

def linguistic_context(seed):

    assoc = st.session_state.shadow_rel

    if seed not in assoc:
        return {"context":"exploration"}

    neighbors = assoc[seed]

    themes = {}
    for w,score in neighbors.items():
        root = w[:4]
        themes[root] = themes.get(root,0)+score

    if not themes:
        return {"context":"vide"}

    context = max(themes, key=themes.get)

    return {
        "context":context,
        "strength":themes[context]
    }

# =====================================================
# S+12 — THINK_GENERATION_ENGINE
# =====================================================

def think(seed, steps=30):
    assoc = st.session_state.shadow_rel

    # Si la graine n'existe pas dans les relations, on ne peut pas lier
    if seed not in assoc:
        return f"Le concept '{seed}' est isolé dans ma structure."

    # Identification du contexte linguistique dominant
    ctx = linguistic_context(seed)

    sent = [seed]
    cur = seed

    # Boucle de génération stochastique (Chaîne de Markov)
    for _ in range(steps):
        nxt = assoc.get(cur)
        if not nxt:
            break
        
        # Sélection pondérée par les scores de fréquences apprises
        w = list(nxt.keys())
        p = np.array(list(nxt.values()), dtype=float)
        p = p ** (1/temp)
        p = p / p.sum()

        cur = np.random.choice(w, p=p)
        sent.append(cur)

    # Mise en forme de la pensée
    sentence = " ".join(sent).capitalize() + "."
    return f"**[Contexte : {ctx['context']}]** {sentence}"

# =====================================================
# S+13 — COGNITIVE_METRICS
# =====================================================

def semantic_coherence():

    concepts=len(st.session_state.shadow_frag)
    assoc=len(st.session_state.shadow_rel)

    return round(min(100,(assoc/max(concepts,1))*10),2)

# =====================================================
# S+14 — AUTO_DIAGNOSTIC_SYSTEM
# =====================================================

def diagnose():
    cortex = st.session_state.shadow_cortex
    density = association_density()

    # Diagnostic basé sur l'activité récente et la densité du réseau
    if cortex["new_today"] < 10:
        return "🧠 Oracle en attente d'assimilation massive."

    if density < 2:
        return "🧠 Réseau en cours de structuration initiale."

    if density > 5:
        return "🧠 Sagesse Titan active : Résonance émergente détectée."

    return "🧠 Absorption de bibliothèques en cours."

# =====================================================
# S+15B — DIRECT_INGESTION_ENGINE (Local)
# =====================================================

import glob

SOURCE_DIR = "source_data"
os.makedirs(SOURCE_DIR, exist_ok=True)

st.subheader("📁 Ingestion Haute Vélocité")

if st.button("Scanner le dossier source_data"):
    # Récupérer tous les fichiers compatibles
    files = glob.glob(f"{SOURCE_DIR}/*.txt") + glob.glob(f"{SOURCE_DIR}/*.docx")
    
    if not files:
        st.warning("Le dossier 'source_data' est vide.")
    else:
        progress_bar = st.progress(0)
        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            
            with st.status(f"Assimilation de {file_name}...", expanded=False):
                if file_name.endswith(".docx"):
                    # On ouvre le fichier en mode binaire local
                    with open(file_path, "rb") as f:
                        text = from docx import Document

def read_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

      else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                
                n = learn(text)
                st.write(f"Succès : {n} fragments extraits.")
            
            progress_bar.progress((i + 1) / len(files))
        
        st.success(f"Traitement terminé : {len(files)} fichiers assimilés.")
        st.rerun()

# =====================================================
# S+15 — USER_DIALOG_INTERFACE
# =====================================================

st.title("🧠 ORACLE S+")

ctx = st.session_state.shadow_cortex

# --- Dashboard ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitalité", round(ctx["VS"], 2))
c2.metric("Age", round(ctx["age"], 6)) 
c3.metric("Densité", association_density())
c4.metric("Cohérence", semantic_coherence())

st.info(diagnose())

# --- Module d'Apprentissage ---
with st.expander("📥 Ingestion de Données"):
    uploaded = st.file_uploader("Fichier cible", type=["txt","csv","docx","pdf"])
    if uploaded:
        if uploaded.name.endswith(".docx"):
            text = read_docx(uploaded)
        else:
            text = uploaded.getvalue().decode("utf-8","ignore")
        n = learn(text)
        st.success(f"{n} unités intégrées au Cortex.")

# --- Espace de Dialogue ---
st.subheader("💬 Interface de Résonance")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Entrez votre phrase ou concept..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ANALYSE DE LA PHRASE (Coordination avec S+12)
    tokens = tokenize(prompt)
    
    if tokens:
        # On cherche le mot de la phrase le plus "connu" (score le plus haut)
        df_frag = st.session_state.shadow_frag
        known_tokens = df_frag[df_frag["fragment"].isin(tokens)]
        
        if not known_tokens.empty:
            # On prend le mot avec le plus grand 'count' comme point de départ
            best_seed = known_tokens.loc[known_tokens["count"].idxmax(), "fragment"]
            
            # L'IA pré-calcule le lien le plus fort
            seed = prethink(best_seed)
            response = think(seed)
        else:
            # Si aucun mot n'est connu, l'IA tente une exploration sauvage
            response = "Mes fragments actuels ne me permettent pas de lier ces concepts."
        
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Système : Flux de caractères insuffisant.")

# =====================================================
# S+16 — STREAMLIT_UI_RENDER
# =====================================================

st.caption(
    f"Temps cognitif : {round(st.session_state.cognitive_time,2)} s"
)
spectral_ui(
    st.session_state.shadow_cortex,
    st.session_state.shadow_frag["fragment"].tolist()
)

# CODE DE CONVERSION UNIQUE
if st.button("Convertir l'âge en échelle Titan"):
    cortex = load_json(FILES["cortex"])
    # Si l'âge est encore en millions (ancienne version)
    if cortex["age"] > 1000:
        cortex["age"] = cortex["age"] / 250_000_000
        save_json(FILES["cortex"], cortex)
        st.success(f"Conversion réussie ! Nouvel âge : {cortex['age']}")
        st.rerun()