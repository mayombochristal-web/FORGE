import streamlit as st
import numpy as np
import hashlib
import plotly.express as px
import networkx as nx
from scipy.special import zeta

# ============================================================
# CONFIG UI
# ============================================================

st.set_page_config(
    page_title="🌌 TTU Riemannium Engine",
    layout="wide"
)

st.title("🌌 Monde Sans Temps — TTU Riemannium")

# ============================================================
# SESSION STATE
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = []

if "phase" not in st.session_state:
    st.session_state.phase = "FLUIDE"

# ============================================================
# 🔐 RIEMANN CORE
# ============================================================

def riemann_signature(data: str):
    """ Génère un identifiant unique basé sur Zeta """
    h = hashlib.sha256(data.encode()).hexdigest()
    val = int(h[:8], 16)
    return np.log(val + 1)

def spectral_density(x):
    """ Approximation densité via fonction zêta """
    return np.abs(zeta(0.5 + 1j * x))

# ============================================================
# ⚙️ MOTEUR Φ = (M, C, D)
# ============================================================

def compute_phi(text):
    M = len(st.session_state.memory)

    # Cohérence = similarité simple
    coherence = sum([len(set(text) & set(m)) for m in st.session_state.memory]) + 1

    # Dissipation = bruit (aléatoire)
    D = np.random.rand() * 10

    return M, coherence, D

# ============================================================
# 🌊 PHASE MANAGER
# ============================================================

def get_phase(C, D):
    ratio = C / (D + 1e-5)

    if ratio > 3:
        return "CRISTAL ❄️"
    elif ratio > 1.5:
        return "FLUIDE 💧"
    elif ratio > 0.7:
        return "VAPEUR 🌫️"
    else:
        return "PLASMA ⚡"

# ============================================================
# 🧠 ORACLE (TRANSFORMATION DU CHAOS)
# ============================================================

def oracle_response(text, C, D):
    if C > D:
        return f"🧠 Résonance stable : {text}"
    else:
        return f"⚡ Chaos transformé → idée : {text[::-1]}"

# ============================================================
# UI INPUT
# ============================================================

user_input = st.text_input("💬 Injecte une idée dans le Riemannium")

if st.button("⚡ Générer") and user_input:

    M, C, D = compute_phi(user_input)
    phase = get_phase(C, D)

    st.session_state.phase = phase
    response = oracle_response(user_input, C, D)

    st.session_state.memory.append(user_input)

    # ========================================================
    # AFFICHAGE Φ
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric("Mémoire (M)", M)
    col2.metric("Cohérence (C)", round(C, 2))
    col3.metric("Dissipation (D)", round(D, 2))

    st.subheader(f"🌡️ Phase actuelle : {phase}")
    st.success(response)

# ============================================================
# 📊 VISUALISATION SPECTRALE
# ============================================================

st.subheader("📊 Spectre du Riemannium")

x = np.linspace(0, 50, 200)
y = spectral_density(x)

fig = px.line(x=x, y=y, title="Densité spectrale (approximation ζ)")
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🌐 GRAPHE MÉMOIRE (HOLOGRAPHIQUE)
# ============================================================

st.subheader("🧬 Mémoire Holographique")

if len(st.session_state.memory) > 1:

    G = nx.Graph()

    for i, m1 in enumerate(st.session_state.memory):
        for j, m2 in enumerate(st.session_state.memory):
            if i != j:
                weight = len(set(m1) & set(m2))
                if weight > 0:
                    G.add_edge(m1, m2, weight=weight)

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []

    for node in G.nodes():
        x0, y0 = pos[node]
        node_x.append(x0)
        node_y.append(y0)

    fig2 = px.scatter(
        x=node_x,
        y=node_y,
        text=list(G.nodes()),
        title="Réseau de cohérence"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 🧊 MODE CRISTAL (ARCHIVES)
# ============================================================

if "CRISTAL" in st.session_state.phase:

    st.subheader("🔐 Coffre Cristal (lecture seule)")
    for m in st.session_state.memory:
        st.code(m)

# ============================================================
# 🌫️ MODE VAPEUR (SANDBOX)
# ============================================================

if "VAPEUR" in st.session_state.phase:

    st.subheader("🧪 Sandbox créatif")

    test = st.text_area("Expérimente une idée instable")

    if st.button("Test Sandbox"):
        st.write("Résultat instable :", test[::-1])

# ============================================================
# ⚡ MODE PLASMA (CALCUL)
# ============================================================

if "PLASMA" in st.session_state.phase:

    st.subheader("⚡ Mode Plasma — Calcul brut")

    val = st.slider("Injecter énergie", 0.0, 100.0, 10.0)
    result = spectral_density(val)

    st.metric("Résultat énergétique", result)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("TTU Riemannium Engine — Monde Sans Temps actif")