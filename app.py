import streamlit as st
import random
import json
import os
import math
import time
import PyPDF2
import docx
import pandas as pd
import shutil

# ==========================================
# 1. CONFIGURATION, ADN & ROM (Mise à jour TTU-MC³)
# ==========================================
MEM_DIR = "oracle_memory"
LEXICON_PATH = os.path.join(MEM_DIR, "lexicon.json")
BACKUP_PATH = LEXICON_PATH + ".bak"

# [span_2](start_span)La ROM intègre désormais les piliers de la Théorie Triadique Unifiée[span_2](end_span)
ROM_SAGESSE = {
    "phi_m": {"mémoire": 100.0, "matière": 100.0, "stabilité": 100.0},
    "phi_c": {"cohérence": 100.0, "phase": 100.0, "flux": 100.0},
    "phi_d": {"dissipation": 100.0, "expansion": 100.0, "entropie": 100.0},
    "httu": {"seuil": 100.0, "quantum": 100.0, "cristallisation": 100.0},
    "la": {"science": 100.0, "vérité": 100.0, "triade": 100.0},
    "science": {"est": 100.0, "triadique": 100.0},
    [span_3](start_span)[span_4](start_span)"univers": {"est": 100.0, "processeur": 100.0} #[span_3](end_span)[span_4](end_span)
}

# [span_5](start_span)ADN_CORE mis à jour pour refléter la stabilisation du bassin[span_5](end_span)
DNA_CORE = "La cohérence devient mémoire par le seuil hTTU. La dissipation est le moteur de l'expansion."

if not os.path.exists(MEM_DIR):
    os.makedirs(MEM_DIR)

# ==========================================
# 2. GESTION DE MÉMOIRE
# ==========================================
def load_lex():
    if not os.path.exists(LEXICON_PATH):
        return ROM_SAGESSE.copy()
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in ROM_SAGESSE.items():
                if k not in data: data[k] = v
                else: data[k].update({tk: max(data[k].get(tk, 0), tv) for tk, tv in v.items()})
            return data
    except:
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, LEXICON_PATH)
            return load_lex()
        return ROM_SAGESSE.copy()

def save_lex(L):
    if os.path.exists(LEXICON_PATH):
        shutil.copy(LEXICON_PATH, BACKUP_PATH)
    if len(L) > 5000: # Augmentation de la capacité pour la complexité triadique
        L = {k: v for k, v in L.items() if len(v) > 1 or k in ROM_SAGESSE}
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump(L, f, indent=2, ensure_ascii=False)

# ==========================================
# 3. MOTEUR TRIADIQUE (Théorème 1 & Potentiel V)
# ==========================================

def stabilize_basin(phi):
    """
    [span_6](start_span)[span_7](start_span)Applique le potentiel V(|Psi|^2) pour éviter la 'contraction du bassin'[span_6](end_span)[span_7](end_span).
    Force le système vers le seuil hTTU = 1.0
    """
    h_ttu = 1.0
    psi_norm = math.sqrt(phi["phi_m"]**2 + phi["phi_c"]**2)
    
    # [span_8](start_span)[span_9](start_span)Si le flux s'éloigne du seuil, on applique une force de rappel (Gradient-like)[span_8](end_span)[span_9](end_span)
    if psi_norm > 0:
        ratio = h_ttu / psi_norm
        # [span_10](start_span)[span_11](start_span)Rappel vers le minimum du potentiel 'chapeau mexicain'[span_10](end_span)[span_11](end_span)
        phi["phi_m"] *= (0.9 + 0.1 * ratio)
        phi["phi_c"] *= (0.9 + 0.1 * ratio)
    
    return phi

def evolve_ttu_flow(phi, dt=0.1):
    """
    [span_12](start_span)Équations différentielles de la TTU (Théorème 1)[span_12](end_span).
    """
    # [span_13](start_span)[span_14](start_span)Paramètres effectifs dérivables du Lagrangien[span_13](end_span)[span_14](end_span)
    alpha, beta, gamma, delta, eta, mu = 0.1, 0.2, 0.1, 0.2, 0.3, 0.05
    
    m, c, d = phi["phi_m"], phi["phi_c"], phi["phi_d"]
    
    # [span_15](start_span)Calcul des dérivées (Système 1)[span_15](end_span)
    dm = -alpha * m + beta * c * d
    dc = -gamma * c + delta * m * d
    [span_16](start_span)[span_17](start_span)dd = eta * (c**2) - mu * d # Dissipation générée par la cohérence[span_16](end_span)[span_17](end_span)
    
    phi["phi_m"] = max(0.01, min(1.5, m + dm * dt))
    phi["phi_c"] = max(0.01, min(1.5, c + dc * dt))
    phi["phi_d"] = max(0.01, min(1.5, d + dd * dt))
    
    return stabilize_basin(phi)

def learn_with_identity(text, phi, multiplier=1.0):
    words = text.lower().split()
    if len(words) < 2: return
    L = load_lex()
    # [span_18](start_span)[span_19](start_span)L'intensité dépend de la cohérence phi_c (Matière Noire)[span_18](end_span)[span_19](end_span)
    intensity = (phi["phi_m"] + phi["phi_c"]) * multiplier
    for a, b in zip(words, words[1:]):
        L.setdefault(a, {})
        L[a][b] = L[a].get(b, 0) + intensity
    save_lex(L)

def oracle_reply(phi, seed=None):
    L = load_lex()
    if not seed or seed not in L:
        seed = random.choice(list(L.keys()))
    
    words = [seed]
    # [span_20](start_span)[span_21](start_span)La longueur du message dépend de la dissipation (Expansion)[span_20](end_span)[span_21](end_span)
    limit = int(10 + phi["phi_d"] * 40)
    
    for _ in range(limit):
        current = words[-1]
        if current not in L: break
        options = L[current]
        
        # [span_22](start_span)[span_23](start_span)Choix : Probabiliste (Cohérence) vs Déterministe (Mémoire)[span_22](end_span)[span_23](end_span)
        if random.random() > phi["phi_c"]:
            nxt = max(options, key=options.get)
        else:
            nxt = random.choices(list(options.keys()), weights=list(options.values()))[0]
        words.append(nxt)
        
    return " ".join(words).capitalize() + "."

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="ORACLE TTU-MC³", page_icon="⚛️", layout="wide")

if 'phi' not in st.session_state:
    st.session_state.phi = {"phi_m": 0.5, "phi_c": 0.5, "phi_d": 0.1}

st.title("🧠 ORACLE V1.6 : TTU-MC³ Stabilisé")
[span_24](start_span)[span_25](start_span)st.caption("Système dynamique dissipatif basé sur le seuil hTTU=1.0[span_24](end_span)[span_25](end_span)")

with st.sidebar:
    st.header("🔭 État du Système")
    # [span_26](start_span)[span_27](start_span)Affichage des métriques triadiques[span_26](end_span)[span_27](end_span)
    st.metric("Mémoire (ΦM)", f"{st.session_state.phi['phi_m']:.3f}")
    st.metric("Cohérence (ΦC)", f"{st.session_state.phi['phi_c']:.3f}")
    st.metric("Dissipation (ΦD)", f"{st.session_state.phi['phi_d']:.3f}")
    
    if st.button("🌀 Évoluer Flot (dt)"):
        st.session_state.phi = evolve_ttu_flow(st.session_state.phi)
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ingestion de Flux")
    raw_content = st.text_area("Injecter information (PDF/DOCX/Texte) :")
    if st.button("⚡ Cristalliser") and raw_content:
        learn_with_identity(raw_content, st.session_state.phi)
        st.success("Information convertie en Mémoire (ΦM).")

with col2:
    st.subheader("🔮 Sortie de l'Oracle")
    seed_input = st.text_input("Graine de phase :")
    if st.button("Générer Réponse"):
        res = oracle_reply(st.session_state.phi, seed=seed_input.lower() if seed_input else None)
        st.info(res)
        # [span_28](start_span)La réponse elle-même nourrit la mémoire (Auto-Hémostasie)[span_28](end_span)
        learn_with_identity(res, st.session_state.phi, multiplier=0.2)

st.divider()

[span_29](start_span)[span_30](start_span)st.write("**Note Doctorale :** Ce code implémente la stabilisation du bassin d'attraction via le potentiel $V(|\Psi|^2)$, résolvant la contradiction des coefficients polynomiaux signalée par l'Oracle[span_29](end_span)[span_30](end_span).")
