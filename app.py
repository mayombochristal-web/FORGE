# =====================================================
# 🧠 ORACLE V12 — ORACLE IMMORTEL
# Shadow Memory + TTU Cognitive Engine
# Compatible V11 (upgrade non destructif)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

# ---------- SCIENTIFIC MODULES ----------
try:
    from scipy.signal import stft, coherence
    from scipy.linalg import eigvals
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except:
    SPECTRAL_AVAILABLE = False
    # On définit des fonctions factices pour éviter les erreurs
    def stft(*args, **kwargs):
        raise ImportError("scipy.signal.stft non disponible")
    # On ne peut pas vraiment définir plt, mais on ne l'utilisera pas
# ---------- CONFIG ----------
st.set_page_config(page_title="ORACLE IMMORTEL", layout="wide")

MEM="oracle_memory"
os.makedirs(MEM,exist_ok=True)

FILES={
 "fragments":f"{MEM}/fragments.csv",
 "concepts":f"{MEM}/concepts.csv",
 "relations":f"{MEM}/relations.json",
 "intentions": f"{MEM}/intentions.csv",
 "cortex":f"{MEM}/cortex.json"
}

ZIP_PATH="oracle_memory.zip"

# =====================================================
# 🔁 MÉMOIRE IMMORTELLE
# =====================================================

def save_brain_zip():
    with zipfile.ZipFile(ZIP_PATH,"w") as z:
        for f in FILES.values():
            if os.path.exists(f):
                z.write(f)

def load_brain_zip():
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH,"r") as z:
            z.extractall(MEM)

load_brain_zip()

# =====================================================
# INIT
# =====================================================

def init():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment","count"]).to_csv(FILES["fragments"],index=False)
    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept", "weight"]).to_csv(FILES["concepts"], index=False)
    if not os.path.exists(FILES["intentions"]):
        pd.DataFrame(columns=["intent", "count"]).to_csv(FILES["intentions"], index=False)
    if not os.path.exists(FILES["relations"]):
        json.dump({},open(FILES["relations"],"w"))

    if not os.path.exists(FILES["cortex"]):
        json.dump({
            "age":0,
            "VS":12,
            "timeline":[],
            "temperature":0.4,
            "IS":1.0
        },open(FILES["cortex"],"w"))

init()

# =====================================================
# LOADERS
# =====================================================

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------------------------------------------
# LAZY CSV LOADER (ANTI-RAM EXPLOSION)
# --------------------------------------------------

def lazy_csv(path, chunksize=5000):
    if not os.path.exists(path):
        return pd.DataFrame()

    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)

# --------------------------------------------------
# DATAFRAME SAVE SAFE
# --------------------------------------------------

def save_csv(df, path):
    df.reset_index(drop=True).to_csv(path, index=False)

# --------------------------------------------------
# UNIVERSAL FILE READER (IA FOOD ENGINE)
# --------------------------------------------------

def read_any_file(file):
    """
    Lecture universelle IA :
    TXT, CSV, XLSX, DOCX, PDF, ZIP récursif
    """

    name = file.name.lower()
    all_text = ""

    try:

        # ---------- ZIP ----------
        if name.endswith(".zip"):

            with zipfile.ZipFile(file) as z:
                for zname in z.namelist():

                    if zname.startswith("__MACOSX"):
                        continue

                    with z.open(zname) as zf:

                        class PseudoFile:
                            def __init__(self, content, filename):
                                self.content = content
                                self.name = filename
                            def read(self):
                                return self.content

                        all_text += read_any_file(
                            PseudoFile(zf.read(), zname)
                        ) + "\n"

            return all_text

        # ---------- TXT ----------
        if name.endswith(".txt"):
            return file.read().decode("utf-8", "ignore")

        # ---------- CSV ----------
        elif name.endswith(".csv"):
            return pd.read_csv(file).to_string()

        # ---------- EXCEL ----------
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file).to_string()

        # ---------- DOCX ----------
        elif name.endswith(".docx"):
            doc_bin = io.BytesIO(file.read())

            with zipfile.ZipFile(doc_bin) as z:
                xml_content = z.read("word/document.xml")

            tree = ET.fromstring(xml_content)

            paragraphs = [
                node.text
                for node in tree.iter()
                if node.tag.endswith("t") and node.text
            ]

            return " ".join(paragraphs)

        # ---------- PDF (fallback stable) ----------
        elif name.endswith(".pdf"):
            raw = file.read().decode("latin-1", "ignore")

            return re.sub(
                r"[^\x20-\x7EàâéèêëîïôùûüœÇ]+",
                " ",
                raw
            )

    except Exception as e:
        return f"[Erreur lecture {name}] {str(e)}"

    return all_text

# --------------------------------------------------
# SPECIALIZED LOADERS FOR ORACLE MEMORY
# --------------------------------------------------

def load_frag(FILES):
    return lazy_csv(FILES["fragments"])

def save_frag(df, FILES):
    save_csv(df, FILES["fragments"])

def load_concepts(FILES):
    return lazy_csv(FILES["concepts"])

def save_concepts(df, FILES):
    save_csv(df, FILES["concepts"])

# =====================================================
# SESSION SHADOW
# =====================================================
def sync_shadow():
    if "shadow_loaded" not in st.session_state:
        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_concepts = load_concepts().copy()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])
        st.session_state.shadow_loaded = True

sync_shadow()
# --------------------------------------------------
# MÉTRIQUES RAPIDES
# --------------------------------------------------
def association_density_fast():
    assoc = st.session_state.shadow_rel
    links = sum(len(v) for v in assoc.values())
    vocab = len(assoc)
    return round(links / max(vocab, 1), 2)

def semantic_coherence_fast():
    concepts = len(st.session_state.shadow_concepts)
    assoc = len(st.session_state.shadow_rel)
    return round(min(100, (assoc / max(concepts, 1)) * 10), 2)
# =====================================================
# NLP
# =====================================================

def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]"," ",t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w)>1]

def semantic_energy(word,freq):
    return 1/(1+freq)
# --------------------------------------------------
# LECTURE DES FICHIERS UPLOADÉS
# --------------------------------------------------
def read_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            return file.read().decode("utf-8", "ignore")
        if name.endswith(".csv"):
            return pd.read_csv(file).to_string()
        if name.endswith(".xlsx"):
            return pd.read_excel(file).to_string()
        if name.endswith(".docx"):
            doc = zipfile.ZipFile(io.BytesIO(file.read()))
            xml = doc.read("word/document.xml")
            tree = ET.fromstring(xml)
            return " ".join(t.text for t in tree.iter() if t.text)
        if name.endswith(".pdf"):
            return file.read().decode("latin-1", "ignore")
    except:
        return ""
    return ""

# --------------------------------------------------
# APPRENTISSAGE (avec timeline pour l'analyse spectrale)
# --------------------------------------------------
def learn(text):
    words = tokenize(text)
    if not words:
        return 0

    # Mise à jour des fragments
    df = st.session_state.shadow_frag.copy()
    counts = Counter(words)
    for w, c in counts.items():
        mask = df["fragment"] == w
        if mask.any():
            df.loc[mask, "count"] += c
        else:
            df = pd.concat([df, pd.DataFrame([[w, c]], columns=df.columns)], ignore_index=True)
    save_frag(df)
    st.session_state.shadow_frag = df

    # Mise à jour des associations (bigrammes)
    assoc = st.session_state.shadow_rel
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        assoc.setdefault(a, {})
        assoc[a][b] = assoc[a].get(b, 0) + 2
    save_json(FILES["relations"], assoc)

    # Mise à jour du cortex (âge, vitalité, timeline)
    cortex = st.session_state.shadow_cortex
    today = str(datetime.date.today())
    if cortex.get("last_day") != today:
        cortex["new_today"] = 0
        cortex["last_day"] = today
    cortex["age"] = cortex.get("age", 0) + len(words)
    cortex["new_today"] = cortex.get("new_today", 0) + len(counts)
    cortex["VS"] = 10 + float(np.log1p(cortex["age"]))

    # Ajout à la timeline
    if "timeline" not in cortex:
        cortex["timeline"] = []
    cortex["timeline"].extend(words)

    save_json(FILES["cortex"], cortex)
    st.session_state.shadow_cortex = cortex
    return len(words)

# --------------------------------------------------
# GÉNÉRATION DE TEXTE (pensée)
# --------------------------------------------------
def think(seed, steps=30):
    assoc = st.session_state.shadow_rel
    if seed not in assoc:
        return "Je dois encore apprendre sur ce concept."

    sent = [seed]
    cur = seed
    for _ in range(steps):
        nxt = assoc.get(cur)
        if not nxt:
            break
        w = list(nxt.keys())
        p = np.array(list(nxt.values()), dtype=float)
        s = p.sum()
        if s == 0:
            break
        p = p / s
        cur = np.random.choice(w, p=p)
        sent.append(cur)
    return " ".join(sent).capitalize() + "."

# --------------------------------------------------
# AUTO-DIAGNOSTIC
# --------------------------------------------------
def diagnose():
    cortex = st.session_state.shadow_cortex
    density = association_density_fast()
    if cortex.get("new_today", 0) < 20:
        return "🧠 J'ai besoin de nouvelles connaissances."
    if density < 1.5:
        return "🧠 Donne-moi des textes plus longs."
    if density > 4:
        return "🧠 Mon raisonnement commence à émerger."
    return "🧠 Apprentissage actif."

# --------------------------------------------------
# FONCTIONS POUR L'ANALYSE SPECTRALE (si disponibles)
# --------------------------------------------------
def build_signal_from_timeline(word):
    """
    Construit un signal binaire (1 si le mot apparaît, 0 sinon) à partir de la timeline.
    """
    cortex = st.session_state.shadow_cortex
    timeline = cortex.get("timeline", [])
    if not timeline:
        return np.array([])
    signal = np.array([1 if w == word else 0 for w in timeline])
    return signal

def spectral_analysis(word, nperseg=256):
    """
    Calcule la STFT du signal binaire du mot et retourne les résultats + figures.
    Nécessite scipy et matplotlib.
    """
    if not SPECTRAL_AVAILABLE:
        st.error("Les bibliothèques scipy et/ou matplotlib ne sont pas installées. Impossible de faire l'analyse spectrale.")
        return None

    signal = build_signal_from_timeline(word)
    if len(signal) < nperseg:
        return {"error": f"Signal trop court (taille={len(signal)}). Augmentez la quantité de textes ou réduisez nperseg."}

    fs = 1.0 # 1 échantillon par mot
    f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg//2)

    # Figure 1 : spectrogramme d'amplitude
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud')
    ax1.set_ylabel('Fréquence [cycles/mot]')
    ax1.set_xlabel('Temps [mot]')
    ax1.set_title(f'Spectrogramme du mot "{word}"')

    # Extraction de la fréquence dominante (moyenne temporelle)
    mean_amp = np.mean(np.abs(Zxx), axis=1)
    idx_max = np.argmax(mean_amp[1:]) + 1 # on ignore la composante continue
    freq_dominant = f[idx_max]
    phase_at_dominant = np.angle(Zxx[idx_max, :])
    phase_unwrapped = np.unwrap(phase_at_dominant)

    # Estimation de l'amortissement alpha (largeur de raie)
    peak_amp = mean_amp[idx_max]
    half_amp = peak_amp / np.sqrt(2) # -3dB
    left_idx = np.where(mean_amp[:idx_max] <= half_amp)[0]
    right_idx = np.where(mean_amp[idx_max:] <= half_amp)[0]
    if len(left_idx) > 0 and len(right_idx) > 0:
        f_left = f[left_idx[-1]]
        f_right = f[idx_max + right_idx[0]]
        bandwidth = f_right - f_left
        alpha = bandwidth / 2
    else:
        alpha = 0.0

    omega = 2 * np.pi * freq_dominant

    # Figure 2 : phase déroulée à la fréquence dominante
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, phase_unwrapped, 'b-')
    ax2.set_xlabel('Temps [mot]')
    ax2.set_ylabel('Phase déroulée [rad]')
    ax2.set_title(f'Phase à f = {freq_dominant:.4f} cycles/mot')

    # Mesure de linéarité de la phase (ajustement linéaire)
    if len(t) > 1:
        coeffs = np.polyfit(t, phase_unwrapped, 1)
        phase_trend = np.polyval(coeffs, t)
        residuals = phase_unwrapped - phase_trend
        linearity = 1 - np.std(residuals) / (np.std(phase_unwrapped) + 1e-10)
    else:
        linearity = 0

    results = {
        "omega": omega,
        "alpha": alpha,
        "lambda": complex(-alpha, omega),
        "freq_dominant": freq_dominant,
        "linearity": linearity,
        "signal_length": len(signal),
        "nperseg": nperseg
    }
    return {"results": results, "figures": (fig1, fig2)}

# --------------------------------------------------
# FONCTION DE TÉLÉCHARGEMENT DES DONNÉES
# --------------------------------------------------
def get_download_data():
    """
    Prépare un dictionnaire contenant toutes les données de l'IA pour téléchargement.
    """
    data = {
        "fragments": st.session_state.shadow_frag.to_dict(orient="records"),
        "concepts": st.session_state.shadow_concepts.to_dict(orient="records"),
        "relations": st.session_state.shadow_rel,
        "cortex": st.session_state.shadow_cortex
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

# --------------------------------------------------
# INTERFACE STREAMLIT
# --------------------------------------------------
st.title("🧠 ORACLE V11 — SHADOW STATE + ANALYSE SPECTRALE V1")

ctx = st.session_state.shadow_cortex

# Métriques en haut
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitalité Spectrale", round(ctx.get("VS", 12), 2))
c2.metric("Âge Cognitif", ctx.get("age", 0))
c3.metric("Densité Associative", association_density_fast())
c4.metric("Cohérence %", semantic_coherence_fast())

st.info(diagnose())

# --------------------------------------------------
# SECTION D'APPRENTISSAGE
# --------------------------------------------------
st.subheader("📥 Nourrir l'IA")
file = st.file_uploader("Nourriture cognitive", type=["txt", "csv", "pdf", "docx", "xlsx"])
if file:
    text = read_file(file)
    n = learn(text)
    st.success(f"{n} unités cognitives assimilées")

# --------------------------------------------------
# SECTION DE CHAT (pensée)
# --------------------------------------------------
st.subheader("💬 Dialogue cognitif")
prompt = st.text_input("Intention")
if st.button("Penser"):
    tokens = tokenize(prompt)
    if not tokens:
        st.warning("Entre une phrase valide.")
    else:
        st.write("### Réponse")
        st.write(think(tokens[0]))

# --------------------------------------------------
# SECTION D'ANALYSE SPECTRALE (si disponible)
# --------------------------------------------------
st.subheader("🔬 Analyse Spectrale")

if not SPECTRAL_AVAILABLE:
    st.warning("⚠️ Les bibliothèques `scipy` et/ou `matplotlib` ne sont pas installées. L'analyse spectrale est désactivée. Pour l'activer, installez-les avec : `pip install scipy matplotlib`.")
    st.info("Vous pouvez néanmoins télécharger les données brutes de l'IA pour une analyse externe (voir section suivante).")
else:
    with st.expander("Voir l'analyse spectrale d'un mot"):
        fragments = st.session_state.shadow_frag["fragment"].tolist()
        if fragments:
            word_to_analyze = st.selectbox("Choisissez un mot", fragments)
            nperseg = st.slider("Taille de la fenêtre STFT", min_value=32, max_value=512, value=128, step=32)

            if st.button("Lancer l'analyse"):
                with st.spinner("Calcul en cours..."):
                    output = spectral_analysis(word_to_analyze, nperseg=nperseg)
                    if output is None:
                        # L'erreur a déjà été affichée dans la fonction
                        pass
                    elif "error" in output:
                        st.error(output["error"])
                    else:
                        res = output["results"]
                        fig1, fig2 = output["figures"]

                        # Affichage des métriques
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Fréquence (cycles/mot)", f"{res['freq_dominant']:.4f}")
                        col2.metric("Omega (rad/mot)", f"{res['omega']:.4f}")
                        col3.metric("Alpha (amort.)", f"{res['alpha']:.4f}")
                        col4.metric("Linéarité phase", f"{res['linearity']:.2f}")

                        # Graphiques
                        st.pyplot(fig1)
                        st.pyplot(fig2)

                        # Interprétation simple
                        if res['linearity'] > 0.8:
                            st.success("La phase est très linéaire → oscillation régulière (mode complexe pur).")
                        elif res['linearity'] < 0.3:
                            st.info("Phase non linéaire → modulation ou comportement chaotique.")
                        else:
                            st.warning("Linéarité modérée.")

                        if res['alpha'] < 0.01:
                            st.write("Amortissement très faible → persistance du sens.")
                        elif res['alpha'] > 0.1:
                            st.write("Amortissement élevé → sens éphémère.")
        else:
            st.info("Aucun mot disponible. Commencez par nourrir l'IA.")

# --------------------------------------------------
# SECTION DE TÉLÉCHARGEMENT DES DONNÉES
# --------------------------------------------------
st.subheader("📤 Télécharger les données de l'IA")
st.markdown("Exportez toutes les données (fragments, concepts, relations, cortex) au format JSON pour une contre-expertise externe.")

if st.button("Préparer l'export"):
    data_json = get_download_data()
    st.download_button(
        label="📥 Télécharger oracle_data.json",
        data=data_json,
        file_name=f"oracle_data_{datetime.date.today()}.json",
        mime="application/json"
    )
# =====================================================
# LEARN (NGRAM + ENERGY TIMELINE)
# =====================================================

def learn(text,n=3):
    words=tokenize(text)
    if not words:
        return 0

    df=st.session_state.shadow_frag.copy()
    counts=Counter(words)

    for w,c in counts.items():
        mask=df["fragment"]==w
        if mask.any():
            df.loc[mask,"count"]+=c
        else:
            df=pd.concat([df,pd.DataFrame([[w,c]],columns=df.columns)])

    df.reset_index(drop=True,inplace=True)
    df.to_csv(FILES["fragments"],index=False)
    st.session_state.shadow_frag=df

    assoc=st.session_state.shadow_rel

    # NGRAMS
    for i in range(len(words)-n):
        key=" ".join(words[i:i+n-1])
        nxt=words[i+n-1]
        assoc.setdefault(key,{})
        assoc[key][nxt]=assoc[key].get(nxt,0)+1

    save_json(FILES["relations"],assoc)

    # ENERGY TIMELINE
    cortex=st.session_state.shadow_cortex

    for w in words:
        freq=int(df[df.fragment==w]["count"].iloc[0])
        energy=semantic_energy(w,freq)
        cortex["timeline"].append({"w":w,"e":energy})

    cortex["age"]+=len(words)
    cortex["VS"]=10+np.log1p(cortex["age"])

    # sincerity index
    cortex["IS"]=np.mean([t["e"] for t in cortex["timeline"][-200:]])

    save_json(FILES["cortex"],cortex)
    save_brain_zip()

    return len(words)

# =====================================================
# THINK (TEMPÉRATURE TTU)
# =====================================================

def think(seed,steps=30):
    assoc=st.session_state.shadow_rel
    T=st.session_state.shadow_cortex["temperature"]

    if seed not in assoc:
        return "Je dois encore apprendre."

    sent=[seed]
    cur=seed

    for _ in range(steps):
        nxt=assoc.get(cur)
        if not nxt: break

        w=list(nxt.keys())
        p=np.array(list(nxt.values()),dtype=float)

        p=p**(1/(T+0.01))
        p=p/p.sum()

        cur=np.random.choice(w,p=p)
        sent.append(cur)

    return " ".join(sent).capitalize()+"."

# =====================================================
# SPECTRAL SIGNAL
# =====================================================

def build_signal(word):
    tl=st.session_state.shadow_cortex["timeline"]
    return np.array([t["e"] if t["w"]==word else 0 for t in tl])

# =====================================================
# CROSS COHERENCE
# =====================================================

def cross_spectrum(w1,w2):
    if not SPECTRAL_AVAILABLE: return None
    s1=build_signal(w1)
    s2=build_signal(w2)
    if len(s1)<64: return None
    f,Cxy=coherence(s1,s2)
    fig,ax=plt.subplots()
    ax.plot(f,Cxy)
    ax.set_title(f"Cohérence {w1}-{w2}")
    return fig

# =====================================================
# EIGEN THOUGHT ANALYSIS
# =====================================================

def eigen_analysis():
    assoc=st.session_state.shadow_rel
    keys=list(assoc.keys())[:200]
    n=len(keys)
    M=np.zeros((n,n))

    for i,k in enumerate(keys):
        for j,v in enumerate(keys):
            if v in assoc.get(k,{}):
                M[i,j]=assoc[k][v]

    vals=eigvals(M)
    return np.real(vals)

# =====================================================
# UI
# =====================================================

st.title("🧠 ORACLE V12 — IMMORTEL")

ctx=st.session_state.shadow_cortex

c1,c2,c3=st.columns(3)
c1.metric("Vitalité",round(ctx["VS"],2))
c2.metric("Age Cognitif",ctx["age"])
c3.metric("Indice Sincérité",round(ctx["IS"],3))

if ctx["IS"]<0.15:
    st.error("⚠️ BRUIT COGNITIF DETECTÉ")

# temperature
ctx["temperature"]=st.slider(
 "Température cognitive",
 0.05,1.5,ctx["temperature"]
)

# LEARN
st.subheader("📥 Nourrir")
file=st.file_uploader("Texte",type=["txt","csv","docx","pdf"])

if file:
    text=file.read().decode("utf-8","ignore")
    n=learn(text)
    st.success(f"{n} unités apprises")

# CHAT
st.subheader("💬 Dialogue")
prompt=st.text_input("Intention")

if st.button("Penser"):
    st.write(think(prompt.lower()))

# CROSS COHERENCE
if SPECTRAL_AVAILABLE:
    st.subheader("🔬 Résonance conceptuelle")
    words=st.session_state.shadow_frag.fragment.tolist()

    if len(words)>2:
        w1=st.selectbox("Mot 1",words,key=1)
        w2=st.selectbox("Mot 2",words,key=2)

        if st.button("Analyser cohérence"):
            fig=cross_spectrum(w1,w2)
            if fig:
                st.pyplot(fig)

# EIGENVALUES
if st.button("Analyse valeurs propres"):
    vals=eigen_analysis()
    st.write("Modes cognitifs dominants :",vals[:10])
