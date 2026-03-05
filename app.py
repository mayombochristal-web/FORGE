# =====================================================
# 🧠 ORACLE V11 — SHADOW STATE + ANALYSE SPECTRALE V1
# TST Ghost Memory + Auto Diagnostic AI + Premiers pas
# vers la Sémantique Spectrale
#
# 📦 Dépendances optionnelles pour l'analyse spectrale :
#    scipy, matplotlib
#    Pour les installer : pip install scipy matplotlib
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, io, zipfile, datetime
import xml.etree.ElementTree as ET
from collections import Counter

# Tentative d'import des modules pour l'analyse spectrale
try:
    from scipy.signal import stft
    import matplotlib.pyplot as plt
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    # On définit des fonctions factices pour éviter les erreurs
    def stft(*args, **kwargs):
        raise ImportError("scipy.signal.stft non disponible")
    # On ne peut pas vraiment définir plt, mais on ne l'utilisera pas

# Configuration de la page
st.set_page_config(page_title="ORACLE V11 SPECTRAL", layout="wide")

# --------------------------------------------------
# CHEMINS ET FICHIERS DE MÉMOIRE
# --------------------------------------------------
MEM = "oracle_memory"
os.makedirs(MEM, exist_ok=True)

FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "concepts": f"{MEM}/concepts.csv",
    "relations": f"{MEM}/relations.json",
    "intentions": f"{MEM}/intentions.csv",
    "cortex": f"{MEM}/cortex.json",
    "feedback": f"{MEM}/feedback.json"  # Pour stocker les feedbacks
}

# --------------------------------------------------
# INITIALISATION DES FICHIERS SI ABSENTS
# --------------------------------------------------
def init():
    if not os.path.exists(FILES["fragments"]):
        pd.DataFrame(columns=["fragment", "count"]).to_csv(FILES["fragments"], index=False)
    if not os.path.exists(FILES["concepts"]):
        pd.DataFrame(columns=["concept", "weight"]).to_csv(FILES["concepts"], index=False)
    if not os.path.exists(FILES["intentions"]):
        pd.DataFrame(columns=["intent", "count"]).to_csv(FILES["intentions"], index=False)
    if not os.path.exists(FILES["relations"]):
        json.dump({}, open(FILES["relations"], "w"))
    if not os.path.exists(FILES["feedback"]):
        json.dump([], open(FILES["feedback"], "w"))
    if not os.path.exists(FILES["cortex"]):
        # On initialise cortex avec une timeline vide
        json.dump({
            "VS": 12,
            "age": 0,
            "new_today": 0,
            "last_day": str(datetime.date.today()),
            "timeline": []  # pour l'analyse spectrale
        }, open(FILES["cortex"], "w"))

init()

# --------------------------------------------------
# FONCTIONS DE CHARGEMENT / SAUVEGARDE
# --------------------------------------------------
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p, d):
    with open(p, "w") as f:
        json.dump(d, f)

def load_frag():
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.reset_index(drop=True).to_csv(FILES["fragments"], index=False)

def load_concepts():
    return pd.read_csv(FILES["concepts"])

def save_concepts(df):
    df.reset_index(drop=True).to_csv(FILES["concepts"], index=False)

# --------------------------------------------------
# CHARGEMENT DES DONNÉES EN SESSION (SHADOW)
# --------------------------------------------------
def sync_shadow():
    if "shadow_loaded" not in st.session_state:
        st.session_state.shadow_frag = load_frag().copy()
        st.session_state.shadow_concepts = load_concepts().copy()
        st.session_state.shadow_rel = load_json(FILES["relations"])
        st.session_state.shadow_cortex = load_json(FILES["cortex"])
        st.session_state.feedback_log = load_json(FILES["feedback"])
        st.session_state.messages = []  # Historique de la conversation
        st.session_state.shadow_loaded = True

sync_shadow()

# --------------------------------------------------
# FILTRAGE DES ARTEFACTS ET STOP WORDS
# --------------------------------------------------
# Liste de mots de transition français (stop words)
STOP_WORDS_FR = {
    'de', 'le', 'la', 'les', 'un', 'une', 'des', 'du', 'et', 'ou', 'mais', 'donc',
    'car', 'ni', 'or', 'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre',
    'votre', 'leur', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'je', 'tu', 'il',
    'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'eux', 'y',
    'en', 'dans', 'sur', 'sous', 'avec', 'sans', 'par', 'pour', 'vers', 'chez',
    'entre', 'depuis', 'pendant', 'avant', 'après', 'dès', 'jusque', 'malgré',
    'selon', 'comme', 'quand', 'lorsque', 'que', 'qui', 'quoi', 'dont', 'où',
    'au', 'aux', 'auquel', 'duquel', 'lequel', 'laquelle', 'lesquels', 'lesquelles',
    'à', 'c', 'd', 'j', 'l', 'm', 'n', 's', 't', 'y', 'ça', 'là', 'très', 'plus',
    'moins', 'aussi', 'autant', 'peu', 'beaucoup', 'trop', 'si', 'oui', 'non',
    'peut-être', 'voilà', 'voici', 'etc'
}

def is_artifact(word):
    """
    Détecte les artefacts techniques :
    - mots trop courts (longueur < 2)
    - mots contenant des chiffres ou des symboles (hors lettres accentuées)
    - mots dans une liste noire (ex: fragments techniques comme 'âk', 'uwwm')
    On peut aussi exclure les mots sans voyelle (pour éliminer le bruit).
    """
    if len(word) < 2:
        return True
    # Vérifie que le mot ne contient que des lettres (y compris accentuées)
    if not re.match(r'^[a-zàâäéèêëïîôöùûüÿœ]+$', word):
        return True
    # Option : exclure les mots sans voyelle (souvent du bruit)
    if not re.search(r'[aeiouyàâäéèêëïîôöùûüÿœ]', word):
        return True
    return False

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

# --------------------------------------------------
# TRAITEMENT DU TEXTE
# --------------------------------------------------
def clean(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean(t).split() if len(w) > 1]

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
# APPRENTISSAGE (avec filtre des artefacts)
# --------------------------------------------------
def learn(text, filter_artifacts=True):
    words = tokenize(text)
    if not words:
        return 0

    if filter_artifacts:
        # On ne garde que les mots qui ne sont pas des artefacts
        words = [w for w in words if not is_artifact(w)]

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
# GÉNÉRATION DE TEXTE (pensée) avec améliorations
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

    # Éviter que le dernier mot soit un stop word
    while sent and sent[-1] in STOP_WORDS_FR:
        sent.pop()
    if not sent:
        return "..."

    # Mettre une majuscule au début et un point à la fin
    phrase = " ".join(sent)
    phrase = phrase[0].upper() + phrase[1:] + "."
    return phrase

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

    fs = 1.0  # 1 échantillon par mot
    f, t, Zxx = stft(signal, fs, window='blackmanharris', nperseg=nperseg, noverlap=nperseg//2)

    # Figure 1 : spectrogramme d'amplitude
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud')
    ax1.set_ylabel('Fréquence [cycles/mot]')
    ax1.set_xlabel('Temps [mot]')
    ax1.set_title(f'Spectrogramme du mot "{word}"')

    # Extraction de la fréquence dominante (moyenne temporelle)
    mean_amp = np.mean(np.abs(Zxx), axis=1)
    idx_max = np.argmax(mean_amp[1:]) + 1  # on ignore la composante continue
    freq_dominant = f[idx_max]
    phase_at_dominant = np.angle(Zxx[idx_max, :])
    phase_unwrapped = np.unwrap(phase_at_dominant)

    # Estimation de l'amortissement alpha (largeur de raie)
    peak_amp = mean_amp[idx_max]
    half_amp = peak_amp / np.sqrt(2)  # -3dB
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
# FONCTIONS DE FEEDBACK
# --------------------------------------------------
def apply_feedback(message_index, is_positive):
    """
    Applique le feedback sur la réponse correspondant à message_index.
    On renforce ou affaiblit les bigrammes de la réponse.
    """
    if message_index < 0 or message_index >= len(st.session_state.messages):
        return
    msg = st.session_state.messages[message_index]
    if msg["role"] != "assistant":
        return
    response_text = msg["content"]
    words = tokenize(response_text)
    if len(words) < 2:
        return

    assoc = st.session_state.shadow_rel
    delta = 1 if is_positive else -1

    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        if a in assoc and b in assoc[a]:
            new_val = assoc[a][b] + delta
            if new_val <= 0:
                # Si ça devient nul ou négatif, on supprime l'entrée
                del assoc[a][b]
                if not assoc[a]:
                    del assoc[a]
            else:
                assoc[a][b] = new_val
    save_json(FILES["relations"], assoc)

    # Enregistrer le feedback dans le journal
    feedback_entry = {
        "timestamp": str(datetime.datetime.now()),
        "message": response_text,
        "feedback": "pertinent" if is_positive else "non pertinent"
    }
    st.session_state.feedback_log.append(feedback_entry)
    save_json(FILES["feedback"], st.session_state.feedback_log)

    # Mettre à jour l'état de la session pour indiquer que le feedback a été donné
    msg["feedback_given"] = True

# --------------------------------------------------
# GESTION DE LA CONVERSATION
# --------------------------------------------------
def send_message():
    user_input = st.session_state.user_input
    if user_input.strip():
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Apprendre du message utilisateur
        learn(user_input, filter_artifacts=True)
        # Générer une réponse
        tokens = tokenize(user_input)
        seed = tokens[0] if tokens else "je"
        # Trouver un seed significatif
        for t in tokens:
            if not is_artifact(t) and t not in STOP_WORDS_FR:
                seed = t
                break
        response = think(seed, steps=20)
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response, "feedback_given": False})
        # Effacer l'input
        st.session_state.user_input = ""

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
        "cortex": st.session_state.shadow_cortex,
        "feedback_log": st.session_state.feedback_log
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
# SECTION D'APPRENTISSAGE (upload de fichiers)
# --------------------------------------------------
st.subheader("📥 Nourrir l'IA")
col1, col2 = st.columns([3, 1])
with col1:
    file = st.file_uploader("Nourriture cognitive", type=["txt", "csv", "pdf", "docx", "xlsx"])
with col2:
    filter_artifacts = st.checkbox("Filtrer les artefacts", value=True, help="Exclure les fragments techniques (bruit) de l'apprentissage")

if file:
    text = read_file(file)
    n = learn(text, filter_artifacts=filter_artifacts)
    st.success(f"{n} unités cognitives assimilées")

# --------------------------------------------------
# SECTION DE CONVERSATION (remplace l'ancien dialogue)
# --------------------------------------------------
st.subheader("💬 Conversation avec l'IA")

# Afficher l'historique des messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"👤 **Vous :** {msg['content']}")
    else:
        col_msg, col_fb = st.columns([5, 1])
        with col_msg:
            st.markdown(f"🤖 **IA :** {msg['content']}")
        with col_fb:
            # Afficher les boutons de feedback seulement si pas déjà donné
            if not msg.get("feedback_given", False):
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍", key=f"up_{i}"):
                        apply_feedback(i, True)
                        st.rerun()
                with col_fb2:
                    if st.button("👎", key=f"down_{i}"):
                        apply_feedback(i, False)
                        st.rerun()
            else:
                st.write("✅")  # Feedback déjà donné

# Input utilisateur
st.text_input("Votre message :", key="user_input", on_change=send_message)

# --------------------------------------------------
# SECTION D'ANALYSE SPECTRALE
# --------------------------------------------------
st.subheader("🔬 Analyse Spectrale")

if not SPECTRAL_AVAILABLE:
    st.warning("⚠️ Les bibliothèques `scipy` et/ou `matplotlib` ne sont pas installées. L'analyse spectrale est désactivée. Pour l'activer, installez-les avec : `pip install scipy matplotlib`.")
    st.info("Vous pouvez néanmoins télécharger les données brutes de l'IA pour une analyse externe (voir section suivante).")
else:
    with st.expander("Voir l'analyse spectrale d'un mot"):
        word_to_analyze = st.text_input("Entrez un mot à analyser", value="").strip().lower()
        if word_to_analyze:
            # Vérification rapide des occurrences
            signal_test = build_signal_from_timeline(word_to_analyze)
            if signal_test.sum() == 0:
                st.warning(f"Le mot '{word_to_analyze}' n'apparaît dans aucun des textes fournis. L'analyse est impossible.")
            else:
                st.info(f"Le mot apparaît {int(signal_test.sum())} fois dans le corpus.")
                nperseg = st.slider("Taille de la fenêtre STFT", min_value=32, max_value=512, value=128, step=32)
                if st.button("Lancer l'analyse"):
                    with st.spinner("Calcul en cours..."):
                        output = spectral_analysis(word_to_analyze, nperseg=nperseg)
                        if output is None:
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
            st.info("Entrez un mot pour lancer l'analyse.")

# --------------------------------------------------
# SECTION DE TÉLÉCHARGEMENT DES DONNÉES
# --------------------------------------------------
st.subheader("📤 Télécharger les données de l'IA")
st.markdown("Exportez toutes les données (fragments, concepts, relations, cortex, feedback) au format JSON pour une contre-expertise externe.")

if st.button("Préparer l'export"):
    data_json = get_download_data()
    st.download_button(
        label="📥 Télécharger oracle_data.json",
        data=data_json,
        file_name=f"oracle_data_{datetime.date.today()}.json",
        mime="application/json"
    )