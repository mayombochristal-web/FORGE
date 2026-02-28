# =====================================================
# 🧠 ORACLE V18 — AUTO-STABILIZING COGNITIVE ENGINE
# Opérateur L Auto-Apprenant avec Minimisation λ
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import datetime
from collections import Counter, defaultdict
from pathlib import Path
# =====================================================
# OPTIONAL IMPORTS (gestion gracieuse)
# =====================================================
try:
from scipy.signal import stft, welch
from scipy.stats import entropy
from scipy.optimize import minimize
SCIPY_AVAILABLE = True
except ImportError:
SCIPY_AVAILABLE = False
try:
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
MATPLOTLIB_AVAILABLE = True
except ImportError:
MATPLOTLIB_AVAILABLE = False
try:
import seaborn as sns
SEABORN_AVAILABLE = True
except ImportError:
SEABORN_AVAILABLE = False
# =====================================================
# CONFIGURATION UNIQUE (CORRECTION PROBLÈME 1)
# =====================================================
st.set_page_config(
page_title="🧠 ORACLE V18 TTU",
layout="wide",
initial_sidebar_state="expanded"
)
# =====================================================
# CONFIGURATION GLOBALE
# =====================================================
class OracleConfig:
"""Configuration centralisée"""
# Chemins
MEMORY_DIR = "oracle_memory"
BACKUP_DIR = "oracle_backups"
# Fichiers de données
FILES = {
"fragments": "fragments.csv",
"concepts": "concepts.csv",
"relations": "relations.json",
"intentions": "intentions.csv",
"cortex": "cortex.json",
"metadata": "metadata.json",
"lyapunov_history": "lyapunov_history.json"
}
# Paramètres d'apprentissage
LEARNING = {
"min_word_length": 2,
"max_word_length": 50,
"ngram_range": (1, 3),
"association_threshold": 2,
"decay_factor": 0.95,
"learning_rate": 1.0,
"L_learning_rate": 0.01, # Nouveau: taux d'apprentissage de L
"lambda_target": -0.1, # Nouveau: λ cible (négatif = stabilité)
"stability_threshold": 0.05
}
# Paramètres de génération
GENERATION = {
"default_temperature": 0.8,
"min_temperature": 0.05,
"max_temperature": 2.0,
"default_steps": 30,
"max_steps": 200,
"top_k": 10,
"nucleus_p": 0.9,
"L_influence": 0.5 # Nouveau: influence de L sur température
}
# Paramètres d'analyse spectrale
SPECTRAL = {
"default_nperseg": 128,
"min_nperseg": 32,
"max_nperseg": 1024,
"overlap_ratio": 0.5,
"window_types": ['hann', 'hamming', 'blackman', 'blackmanharris', 'bartlett'],
"default_window": 'blackmanharris'
}
# Métriques
METRICS = {
"high_density_threshold": 4.0,
"low_density_threshold": 1.5,
"min_daily_intake": 20,
"coherence_scaling": 10,
"vitalite_base": 10.0
}
# Interface
UI = {
"chunk_size": 5000,
"max_display_items": 100,
"chart_height": 400,
"chart_width": 800
}
# =====================================================
# INITIALISATION
# =====================================================
def init_directories():
"""Crée les répertoires nécessaires"""
Path(OracleConfig.MEMORY_DIR).mkdir(exist_ok=True)
Path(OracleConfig.BACKUP_DIR).mkdir(exist_ok=True)
def get_file_path(key):
"""Retourne le chemin complet d'un fichier"""
return os.path.join(OracleConfig.MEMORY_DIR, OracleConfig.FILES[key])
# =====================================================
# JSON IO SAFE (CORRECTION PROBLÈME 2)
# =====================================================
def load_json(path, default=None):
"""Charge un fichier JSON de manière sûre"""
if not os.path.exists(path):
return default if default is not None else {}
try:
with open(path, "r", encoding="utf-8") as f:
return json.load(f)
except (json.JSONDecodeError, IOError):
return default if default is not None else {}
def save_json(path, data):
"""Sauvegarde un fichier JSON"""
with open(path, "w", encoding="utf-8") as f:
json.dump(data, f, ensure_ascii=False, indent=2)
# =====================================================
# CSV IO
# =====================================================
def load_csv_lazy(path, chunksize=None):
"""Charge un CSV par chunks"""
if not os.path.exists(path):
return pd.DataFrame()
chunksize = chunksize or OracleConfig.UI["chunk_size"]
chunks = []
try:
for chunk in pd.read_csv(path, chunksize=chunksize):
chunks.append(chunk)
return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
except (pd.errors.EmptyDataError, IOError):
return pd.DataFrame()
def save_csv(df, path):
"""Sauvegarde un DataFrame en CSV"""
df.reset_index(drop=True).to_csv(path, index=False)
# =====================================================
# INITIALISATION DES FICHIERS
# =====================================================
def init_files():
"""Initialise les fichiers de données s'ils n'existent pas"""
# Fragments
fragments_path = get_file_path("fragments")
if not os.path.exists(fragments_path):
pd.DataFrame(columns=["fragment", "count", "last_seen", "weight"]).to_csv(
fragments_path, index=False
)
# Concepts
concepts_path = get_file_path("concepts")
if not os.path.exists(concepts_path):
pd.DataFrame(columns=["concept", "weight", "category", "created"]).to_csv(
concepts_path, index=False
)
# Intentions
intentions_path = get_file_path("intentions")
if not os.path.exists(intentions_path):
pd.DataFrame(columns=["intent", "count", "last_used", "success_rate"]).to_csv(
intentions_path, index=False
)
# Relations
relations_path = get_file_path("relations")
if not os.path.exists(relations_path):
save_json(relations_path, {})
# Cortex
cortex_path = get_file_path("cortex")
if not os.path.exists(cortex_path):
cortex_data = {
"version": "18.0",
"created": str(datetime.datetime.now()),
"age": 0,
"vitalite_spectrale": OracleConfig.METRICS["vitalite_base"],
"temperature": OracleConfig.GENERATION["default_temperature"],
"new_today": 0,
"last_day": str(datetime.date.today()),
"timeline": [],
"L_timeline": [], # Nouveau: historique de L
"lambda_timeline": [], # Nouveau: historique de λ
"session_history": [],
"total_sessions": 0,
"auto_stabilization_active": True # Nouveau
}
save_json(cortex_path, cortex_data)
# Metadata
metadata_path = get_file_path("metadata")
if not os.path.exists(metadata_path):
metadata = {
"created": str(datetime.datetime.now()),
"version": "18.0",
"capabilities": {
"scipy": SCIPY_AVAILABLE,
"matplotlib": MATPLOTLIB_AVAILABLE,
"seaborn": SEABORN_AVAILABLE,
"auto_learning_L": True
}
}
save_json(metadata_path, metadata)
# Lyapunov history
lyapunov_path = get_file_path("lyapunov_history")
if not os.path.exists(lyapunov_path):
save_json(lyapunov_path, {
"history": [],
"optimal_weights": {},
"last_optimization": None
})
# =====================================================
# TRAITEMENT LINGUISTIQUE
# =====================================================
class LinguisticProcessor:
"""Traitement linguistique avancé"""
@staticmethod
def tokenize(text, min_length=None, max_length=None):
"""Tokenisation avec filtrage"""
min_length = min_length or OracleConfig.LEARNING["min_word_length"]
max_length = max_length or OracleConfig.LEARNING["max_word_length"]
tokens = re.findall(r"[a-zàâäéèêëîïôùûüœç]+", text.lower())
return [t for t in tokens if min_length <= len(t) <= max_length]
@staticmethod
def extract_ngrams(tokens, n=2):
"""Extrait les n-grammes"""
return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
@staticmethod
def phonemes(text):
"""Extraction approximative de phonèmes"""
text = text.lower()
return list(re.sub(r"[^a-zàâäéèêëîïôùûüœç]", "", text))
@staticmethod
def syllables(word):
"""Découpage approximatif en syllabes"""
pattern = r"[bcdfghjklmnpqrstvwxz]*[aeiouyàâäéèêëîïôùûüœ]+"
return re.findall(pattern, word.lower())
# =====================================================
# SHADOW STATE (CORRECTION PROBLÈMES 4 ET 5)
# =====================================================
class ShadowState:
"""Gestion de l'état en mémoire pour performance"""
@staticmethod
def initialize():
"""Charge toutes les données en session_state"""
if "shadow_initialized" in st.session_state:
return
st.session_state.shadow_fragments = load_csv_lazy(get_file_path("fragments"))
st.session_state.shadow_concepts = load_csv_lazy(get_file_path("concepts"))
st.session_state.shadow_intentions = load_csv_lazy(get_file_path("intentions"))
st.session_state.shadow_rel = load_json(get_file_path("relations"))
st.session_state.shadow_cortex = load_json(get_file_path("cortex"))
st.session_state.shadow_metadata = load_json(get_file_path("metadata"))
st.session_state.lyapunov_data = load_json(get_file_path("lyapunov_history"))
st.session_state.shadow_initialized = True
@staticmethod
def persist():
"""Sauvegarde l'état en mémoire vers les fichiers"""
save_csv(st.session_state.shadow_fragments, get_file_path("fragments"))
save_csv(st.session_state.shadow_concepts, get_file_path("concepts"))
save_csv(st.session_state.shadow_intentions, get_file_path("intentions"))
save_json(get_file_path("relations"), st.session_state.shadow_rel)
save_json(get_file_path("cortex"), st.session_state.shadow_cortex)
save_json(get_file_path("metadata"), st.session_state.shadow_metadata)
save_json(get_file_path("lyapunov_history"), st.session_state.lyapunov_data)
@staticmethod
def reload():
"""Recharge les données depuis les fichiers"""
st.session_state.pop("shadow_initialized", None)
ShadowState.initialize()
# =====================================================
# 🧠 OPERATEUR L DYNAMIQUE TTU (NOUVELLE ARCHITECTURE)
# =====================================================
class OperatorL:
"""
"""
@staticmethod
def compute(tokens):
"""
"""
if not tokens:
return {
'L': 0.0,
'lambda': 0.0,
'stability': 'neutral',
'gradient': 0.0,
'divergence': 0.0,
'concept_hits': []
}
relations = st.session_state.shadow_rel
concepts_df = st.session_state.shadow_concepts
energy = 0.0
concept_hits = []
pair_weights = []
# Phase 1: Calcul de l'énergie par paires
for i in range(len(tokens) - 1):
word_a = tokens[i]
word_b = tokens[i + 1]
# Cherche la relation
if word_a in relations and word_b in relations[word_a]:
weight = relations[word_a][word_b]
energy += weight
pair_weights.append(weight)
# Vérifie si concept
if not concepts_df.empty and word_b in concepts_df['concept'].values:
concept_hits.append(word_b)
# Phase 2: Facteur conceptuel
unique_concepts = len(set(concept_hits))
concept_factor = unique_concepts / (len(tokens) + 1)
# Phase 3: Gradient (moyenne des poids)
gradient = np.mean(pair_weights) if pair_weights else 0.0
# Phase 4: Divergence (variance normalisée)
divergence = np.std(pair_weights) / (np.mean(pair_weights) + 1e-9) if pair_weights else 0.0
# Phase 5: Énergie L stabilisée (tanh pour borner)
L_raw = energy * concept_factor
L_value = np.tanh(L_raw)
# Phase 6: Estimation λ (exposant de Lyapunov)
# λ > 0 → divergence (chaos)
# λ < 0 → convergence (stabilité)
# λ ≈ 0 → neutralité
lambda_estimate = gradient - divergence
# Phase 7: Classification de stabilité
if lambda_estimate < -OracleConfig.LEARNING["stability_threshold"]:
stability = "stable"
elif lambda_estimate > OracleConfig.LEARNING["stability_threshold"]:
stability = "chaotic"
else:
stability = "neutral"
return {
'L': float(L_value),
'lambda': float(lambda_estimate),
'stability': stability,
'gradient': float(gradient),
'divergence': float(divergence),
'concept_hits': concept_hits
}
@staticmethod
def auto_adjust_weights(tokens, target_lambda=None):
"""
"""
target_lambda = target_lambda or OracleConfig.LEARNING["lambda_target"]
learning_rate = OracleConfig.LEARNING["L_learning_rate"]
relations = st.session_state.shadow_rel
# Calcul de L actuel
result = OperatorL.compute(tokens)
current_lambda = result['lambda']
# Erreur
error = target_lambda - current_lambda
# Ajustement des poids (descente de gradient)
for i in range(len(tokens) - 1):
word_a = tokens[i]
word_b = tokens[i + 1]
if word_a in relations and word_b in relations[word_a]:
# Gradient: si λ trop élevé → diminuer poids
# si λ trop bas → augmenter poids
adjustment = learning_rate * error
# Mise à jour
new_weight = relations[word_a][word_b] + adjustment
# Clamping (éviter valeurs négatives ou explosives)
relations[word_a][word_b] = max(0.1, min(100.0, new_weight))
# Sauvegarde
st.session_state.shadow_rel = relations
return {
'error': error,
'adjustment': learning_rate * error,
'new_lambda': OperatorL.compute(tokens)['lambda']
}
@staticmethod
def track_evolution(tokens, result):
"""
"""
cortex = st.session_state.shadow_cortex
timestamp = str(datetime.datetime.now())
# Timeline de L
if "L_timeline" not in cortex:
cortex["L_timeline"] = []
cortex["L_timeline"].append({
"time": timestamp,
"L": result['L'],
"tokens": len(tokens),
"stability": result['stability']
})
# Timeline de λ
if "lambda_timeline" not in cortex:
cortex["lambda_timeline"] = []
cortex["lambda_timeline"].append({
"time": timestamp,
"lambda": result['lambda'],
"gradient": result['gradient'],
"divergence": result['divergence']
})
# Limitation (garder 10000 derniers points)
cortex["L_timeline"] = cortex["L_timeline"][-10000:]
cortex["lambda_timeline"] = cortex["lambda_timeline"][-10000:]
st.session_state.shadow_cortex = cortex
# =====================================================
# MOTEUR D'APPRENTISSAGE
# =====================================================
class LearningEngine:
"""Moteur d'apprentissage avec intégration de L"""
@staticmethod
def learn(text, persist=True, auto_stabilize=True):
"""Apprentissage complet depuis un texte"""
tokens = LinguisticProcessor.tokenize(text)
if not tokens:
return {"words": 0, "new_words": 0, "associations": 0, "L": 0.0}
stats = {
"words": len(tokens),
"new_words": 0,
"associations": 0,
"unique_words": len(set(tokens))
}
# Mise à jour des fragments
stats["new_words"] = LearningEngine._update_fragments(tokens)
# Mise à jour des associations
stats["associations"] = LearningEngine._update_associations(tokens)
# Extraction de concepts
LearningEngine._extract_concepts(tokens)
# NOUVEAU: Calcul de L
L_result = OperatorL.compute(tokens)
stats["L"] = L_result['L']
stats["lambda"] = L_result['lambda']
stats["stability"] = L_result['stability']
# NOUVEAU: Auto-stabilisation si activée
if auto_stabilize and st.session_state.shadow_cortex.get("auto_stabilization_active", True):
adjustment = OperatorL.auto_adjust_weights(tokens)
stats["adjustment"] = adjustment
# Tracking de l'évolution
OperatorL.track_evolution(tokens, L_result)
# Mise à jour du cortex
LearningEngine._update_cortex(tokens, stats)
# Sauvegarde
if persist:
ShadowState.persist()
return stats
@staticmethod
def _update_fragments(tokens):
"""Met à jour la table des fragments"""
df = st.session_state.shadow_fragments
counts = Counter(tokens)
new_words = 0
today = str(datetime.date.today())
for word, count in counts.items():
mask = df["fragment"] == word
if mask.any():
df.loc[mask, "count"] += count
df.loc[mask, "last_seen"] = today
df.loc[mask, "weight"] = df.loc[mask, "count"] * OracleConfig.LEARNING["learning_rate"]
else:
new_row = pd.DataFrame([{
"fragment": word,
"count": count,
"last_seen": today,
"weight": count * OracleConfig.LEARNING["learning_rate"]
}])
df = pd.concat([df, new_row], ignore_index=True)
new_words += 1
st.session_state.shadow_fragments = df
return new_words
@staticmethod
def _update_associations(tokens):
"""Met à jour les associations (bigrammes)"""
assoc = st.session_state.shadow_rel
associations_added = 0
for i in range(len(tokens) - 1):
a, b = tokens[i], tokens[i+1]
if a not in assoc:
assoc[a] = {}
assoc[a][b] = assoc[a].get(b, 0) + OracleConfig.LEARNING["association_threshold"]
associations_added += 1
st.session_state.shadow_rel = assoc
return associations_added
@staticmethod
def _update_cortex(tokens, stats):
"""Met à jour le cortex (état global)"""
cortex = st.session_state.shadow_cortex
today = str(datetime.date.today())
if cortex.get("last_day") != today:
cortex["new_today"] = 0
cortex["last_day"] = today
cortex["age"] = cortex.get("age", 0) + stats["words"]
cortex["new_today"] = cortex.get("new_today", 0) + stats["unique_words"]
cortex["vitalite_spectrale"] = OracleConfig.METRICS["vitalite_base"] + float(np.log1p(cortex["age"]))
# Timeline globale
if "timeline" not in cortex:
cortex["timeline"] = []
cortex["timeline"].extend(tokens)
# Limitation mémoire
max_timeline = 100000
if len(cortex["timeline"]) > max_timeline:
cortex["timeline"] = cortex["timeline"][-max_timeline:]
st.session_state.shadow_cortex = cortex
@staticmethod
def _extract_concepts(tokens):
"""Extrait et met à jour les concepts"""
df_concepts = st.session_state.shadow_concepts
freq = Counter(tokens)
threshold = 3
for word, count in freq.items():
if count >= threshold:
mask = df_concepts["concept"] == word
if mask.any():
df_concepts.loc[mask, "weight"] += count
else:
new_concept = pd.DataFrame([{
"concept": word,
"weight": count,
"category": "auto",
"created": str(datetime.date.today())
}])
df_concepts = pd.concat([df_concepts, new_concept], ignore_index=True)
st.session_state.shadow_concepts = df_concepts
# =====================================================
# MOTEUR DE GÉNÉRATION (AVEC INTÉGRATION L)
# =====================================================
class GenerationEngine:
"""Génération de texte avec contrôle par L"""
@staticmethod
def think(seed, steps=None, temperature=None, method="weighted", use_L=True):
"""
"""
steps = steps or OracleConfig.GENERATION["default_steps"]
base_temperature = temperature or st.session_state.shadow_cortex.get(
"temperature", OracleConfig.GENERATION["default_temperature"]
)
assoc = st.session_state.shadow_rel
if seed not in assoc:
return "Je dois encore apprendre ce concept."
sequence = [seed]
current = seed
for step in range(steps):
next_words = assoc.get(current)
if not next_words:
break
# NOUVEAU: Modulation par L
if use_L:
L_result = OperatorL.compute(sequence)
L_value = L_result['L']
# L élevé → température baisse (stabilité)
# L faible → température monte (créativité)
influence = OracleConfig.GENERATION["L_influence"]
adjusted_temp = max(
OracleConfig.GENERATION["min_temperature"],
base_temperature * (1.0 - influence * L_value)
)
else:
adjusted_temp = base_temperature
# Choix du mot suivant
if method == "weighted":
current = GenerationEngine._weighted_choice(next_words, adjusted_temp)
elif method == "top_k":
current = GenerationEngine._top_k_choice(next_words, adjusted_temp)
elif method == "nucleus":
current = GenerationEngine._nucleus_choice(next_words, adjusted_temp)
sequence.append(current)
# Éviter les boucles
if sequence.count(current) > 3:
break
# Tracking final
final_L = OperatorL.compute(sequence)
OperatorL.track_evolution(sequence, final_L)
return " ".join(sequence).capitalize() + "."
@staticmethod
def _weighted_choice(candidates, temperature):
"""Choix pondéré par poids et température"""
words = list(candidates.keys())
weights = np.array(list(candidates.values()), dtype=float)
weights = weights ** (1.0 / (temperature + 0.01))
weights = weights / weights.sum()
return np.random.choice(words, p=weights)
@staticmethod
def _top_k_choice(candidates, temperature):
"""Choix parmi les top-k candidats"""
k = OracleConfig.GENERATION["top_k"]
sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
top_candidates = dict(sorted_candidates[:k])
return GenerationEngine._weighted_choice(top_candidates, temperature)
@staticmethod
def _nucleus_choice(candidates, temperature):
"""Nucleus sampling (top-p)"""
p = OracleConfig.GENERATION["nucleus_p"]
sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
weights = np.array([w for _, w in sorted_candidates])
weights = weights / weights.sum()
cumsum = np.cumsum(weights)
cutoff = np.searchsorted(cumsum, p) + 1
nucleus = dict(sorted_candidates[:cutoff])
return GenerationEngine._weighted_choice(nucleus, temperature)
# =====================================================
# MÉTRIQUES
# =====================================================
class MetricsEngine:
"""Calcul des métriques cognitives"""
@staticmethod
def association_density():
"""Densité du graphe d'associations"""
assoc = st.session_state.shadow_rel
if not assoc:
return 0.0
total_links = sum(len(v) for v in assoc.values())
vocab_size = len(assoc)
return round(total_links / max(vocab_size, 1), 2)
@staticmethod
def semantic_coherence():
"""Cohérence sémantique"""
concepts = len(st.session_state.shadow_concepts)
assoc = len(st.session_state.shadow_rel)
if assoc == 0:
return 0.0
coherence = (concepts / max(assoc, 1)) * OracleConfig.METRICS["coherence_scaling"]
return round(min(100, coherence), 2)
@staticmethod
def vocabulary_diversity():
"""Diversité du vocabulaire (entropie)"""
df = st.session_state.shadow_fragments
if df.empty:
return 0.0
counts = df["count"].values
probs = counts / counts.sum()
return round(entropy(probs), 2)
@staticmethod
def current_lambda():
"""λ actuel moyen"""
cortex = st.session_state.shadow_cortex
lambda_timeline = cortex.get("lambda_timeline", [])
if not lambda_timeline:
return 0.0
recent = lambda_timeline[-100:] # 100 derniers
return round(np.mean([x['lambda'] for x in recent]), 4)
@staticmethod
def stability_score():
"""Score de stabilité (% de temps en mode stable)"""
cortex = st.session_state.shadow_cortex
L_timeline = cortex.get("L_timeline", [])
if not L_timeline:
return 0.0
recent = L_timeline[-1000:]
stable_count = sum(1 for x in recent if x.get('stability') == 'stable')
return round(100 * stable_count / len(recent), 1)
@staticmethod
def diagnose():
"""Diagnostic automatique"""
density = MetricsEngine.association_density()
lambda_val = MetricsEngine.current_lambda()
stability = MetricsEngine.stability_score()
if lambda_val < -0.1 and stability > 70:
return "🟢 Système auto-stabilisé (λ < 0)"
elif lambda_val > 0.1:
return "🔴 Divergence détectée (λ > 0) — auto-correction active"
elif density < OracleConfig.METRICS["low_density_threshold"]:
return "🟡 Réseau faible — besoin d'apprentissage"
else:
return "🔵 Apprentissage actif"
# =====================================================
# ANALYSE SPECTRALE
# =====================================================
class SpectralAnalysis:
"""Analyse spectrale des patterns temporels"""
@staticmethod
def is_available():
return SCIPY_AVAILABLE and MATPLOTLIB_AVAILABLE
@staticmethod
def build_signal(word):
timeline = st.session_state.shadow_cortex.get("timeline", [])
if not timeline:
return np.array([])
return np.array([1 if w == word else 0 for w in timeline])
@staticmethod
def analyze_word(word, nperseg=128, window='blackmanharris'):
"""Analyse spectrale complète d'un mot"""
if not SpectralAnalysis.is_available():
return {"error": "Scipy/matplotlib manquants"}
signal = SpectralAnalysis.build_signal(word)
if len(signal) < nperseg:
return {"error": f"Signal trop court ({len(signal)} < {nperseg})"}
fs = 1.0
noverlap = int(nperseg * OracleConfig.SPECTRAL["overlap_ratio"])
f, t, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
mean_amp = np.mean(np.abs(Zxx), axis=1)
idx_dominant = np.argmax(mean_amp[1:]) + 1
freq_dominant = f[idx_dominant]
omega = 2 * np.pi * freq_dominant
# Amortissement
alpha = SpectralAnalysis._estimate_damping(mean_amp, f, idx_dominant)
# Phase
phase = np.angle(Zxx[idx_dominant, :])
phase_unwrapped = np.unwrap(phase)
linearity = SpectralAnalysis._phase_linearity(t, phase_unwrapped)
# Figures
fig1 = SpectralAnalysis._plot_spectrogram(t, f, Zxx, word)
fig2 = SpectralAnalysis._plot_phase(t, phase_unwrapped, freq_dominant)
results = {
"omega": omega,
"alpha": alpha,
"lambda": complex(-alpha, omega),
"freq_dominant": freq_dominant,
"linearity": linearity,
"signal_length": len(signal)
}
return {"results": results, "figures": (fig1, fig2)}
@staticmethod
def _estimate_damping(amplitudes, frequencies, peak_idx):
peak_amp = amplitudes[peak_idx]
half_power = peak_amp / np.sqrt(2)
left_indices = np.where(amplitudes[:peak_idx] <= half_power)[0]
right_indices = np.where(amplitudes[peak_idx:] <= half_power)[0]
if len(left_indices) > 0 and len(right_indices) > 0:
f_left = frequencies[left_indices[-1]]
f_right = frequencies[peak_idx + right_indices[0]]
return (f_right - f_left) / 2
return 0.0
@staticmethod
def _phase_linearity(time, phase):
if len(time) < 2:
return 0.0
coeffs = np.polyfit(time, phase, 1)
trend = np.polyval(coeffs, time)
residuals = phase - trend
return 1 - np.std(residuals) / (np.std(phase) + 1e-10)
@staticmethod
def _plot_spectrogram(t, f, Zxx, word):
fig, ax = plt.subplots(figsize=(10, 5))
magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
im = ax.pcolormesh(t, f, magnitude_db, shading='gouraud', cmap='viridis')
ax.set_ylabel('Fréquence [cycles/mot]')
ax.set_xlabel('Temps [position]')
ax.set_title(f'Spectrogramme: "{word}"')
plt.colorbar(im, ax=ax, label='Magnitude [dB]')
return fig
@staticmethod
def _plot_phase(t, phase, freq):
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, phase, 'b-', linewidth=1.5)
ax.set_xlabel('Temps [position]')
ax.set_ylabel('Phase [rad]')
ax.set_title(f'Phase @ f={freq:.4f} cycles/mot')
ax.grid(True, alpha=0.3)
return fig
# =====================================================
# LECTURE DE FICHIERS (CORRECTION PROBLÈME 3)
# =====================================================
def read_any_file(uploaded_file):
"""Lit un fichier uploadé et retourne le texte"""
filename = uploaded_file.name.lower()
try:
if filename.endswith('.txt'):
return uploaded_file.read().decode('utf-8', errors='ignore')
elif filename.endswith('.csv'):
df = pd.read_csv(uploaded_file)
return ' '.join(df.astype(str).values.flatten())
elif filename.endswith('.json'):
data = json.load(uploaded_file)
return json.dumps(data)
else:
return uploaded_file.read().decode('utf-8', errors='ignore')
except Exception as e:
st.error(f"Erreur de lecture: {e}")
return ""
# =====================================================
# EXPORT / IMPORT
# =====================================================
class DataExporter:
"""Exportation des données"""
@staticmethod
def export_all():
data = {
"version": "18.0",
"exported": str(datetime.datetime.now()),
"fragments": st.session_state.shadow_fragments.to_dict(orient="records"),
"concepts": st.session_state.shadow_concepts.to_dict(orient="records"),
"relations": st.session_state.shadow_rel,
"cortex": st.session_state.shadow_cortex,
"lyapunov": st.session_state.lyapunov_data
}
return json.dumps(data, indent=2, ensure_ascii=False)
@staticmethod
def import_all(json_data):
try:
data = json.loads(json_data)
st.session_state.shadow_fragments = pd.DataFrame(data.get("fragments", []))
st.session_state.shadow_concepts = pd.DataFrame(data.get("concepts", []))
st.session_state.shadow_rel = data.get("relations", {})
st.session_state.shadow_cortex = data.get("cortex", {})
st.session_state.lyapunov_data = data.get("lyapunov", {})
ShadowState.persist()
return True
except Exception as e:
st.error(f"Erreur d'import: {e}")
return False
# =====================================================
# INTERFACE STREAMLIT
# =====================================================
def main():
"""Interface principale V18"""
# Initialisation
init_directories()
init_files()
ShadowState.initialize()
# CSS
st.markdown("""
""", unsafe_allow_html=True)
# En-tête
st.markdown('<p class="main-header">🧠 ORACLE V18 — AUTO-STABILIZING ENGINE</p>', unsafe_allow_html=True)
# Sidebar
with st.sidebar:
st.header("⚙️ Configuration")
# Auto-stabilisation
auto_stab = st.checkbox(
"Auto-stabilisation active",
value=st.session_state.shadow_cortex.get("auto_stabilization_active", True)
)
st.session_state.shadow_cortex["auto_stabilization_active"] = auto_stab
# Température
temp = st.slider("Température base", 0.1, 2.0, 0.8, 0.1)
st.session_state.shadow_cortex["temperature"] = temp
# Influence de L
l_influence = st.slider("Influence de L", 0.0, 1.0, 0.5, 0.1)
OracleConfig.GENERATION["L_influence"] = l_influence
# Méthode
gen_method = st.selectbox("Méthode", ["weighted", "top_k", "nucleus"])
gen_steps = st.slider("Longueur", 10, 200, 30)
st.divider()
# Actions
st.header("🔧 Actions")
if st.button("💾 Sauvegarder"):
ShadowState.persist()
st.success("✅ Sauvegarde OK")
if st.button("🔄 Recharger"):
ShadowState.reload()
st.success("✅ Rechargement OK")
st.divider()
# Capacités
st.header("🎯 Système")
st.write(f"**Scipy**: {'✅' if SCIPY_AVAILABLE else '❌'}")
st.write(f"**Matplotlib**: {'✅' if MATPLOTLIB_AVAILABLE else '❌'}")
st.write(f"**Auto-Learning**: ✅")
# Métriques
st.header("📊 Tableau de Bord Cognitif")
col1, col2, col3, col4, col5, col6 = st.columns(6)
cortex = st.session_state.shadow_cortex
with col1:
st.metric("Vitalité", f"{cortex.get('vitalite_spectrale', 10):.1f}")
with col2:
st.metric("Âge", cortex.get("age", 0))
with col3:
st.metric("Densité", MetricsEngine.association_density())
with col4:
st.metric("Cohérence %", MetricsEngine.semantic_coherence())
with col5:
lambda_val = MetricsEngine.current_lambda()
st.metric("λ (Lyapunov)", f"{lambda_val:.4f}",
delta="stable" if lambda_val < 0 else "diverge")
with col6:
st.metric("Stabilité %", MetricsEngine.stability_score())
# Diagnostic
st.info(f"**État**: {MetricsEngine.diagnose()}")
st.divider()
# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
"📚 Apprentissage",
"💬 Génération",
"🔬 Analyse Spectrale",
"📈 Évolution de λ",
"🎯 Auto-Stabilisation",
"⚡ Export/Import"
])
# TAB 1: Apprentissage
with tab1:
st.header("📚 Apprentissage Cognitif")
uploaded = st.file_uploader("Charger un fichier", type=["txt", "csv", "json"])
if uploaded:
text = read_any_file(uploaded)
st.text_area("Aperçu", text[:500], height=150)
if st.button("🧠 Apprendre"):
with st.spinner("Apprentissage..."):
stats = LearningEngine.learn(text)
st.success("✅ Apprentissage terminé!")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Mots appris", stats['words'])
col_b.metric("Nouveaux mots", stats['new_words'])
col_c.metric("L final", f"{stats['L']:.3f}")
if 'adjustment' in stats:
st.info(f"Auto-ajustement: Δλ = {stats['adjustment']['error']:.4f}")
# Saisie manuelle
st.subheader("Saisie manuelle")
manual = st.text_area("Texte", height=100)
if st.button("Apprendre (manuel)"):
if manual.strip():
stats = LearningEngine.learn(manual)
st.success(f"✅ {stats['words']} mots | L={stats['L']:.3f}")
# TAB 2: Génération
with tab2:
st.header("💬 Génération Cognitive")
seed = st.text_input("Mot de départ")
use_L = st.checkbox("Modulation par L", value=True)
if st.button("🚀 Générer"):
if seed.strip():
tokens = LinguisticProcessor.tokenize(seed)
if tokens:
result = GenerationEngine.think(
tokens[0],
steps=gen_steps,
temperature=temp,
method=gen_method,
use_L=use_L
)
st.write("### Résultat")
st.write(result)
# Afficher L final
final_L = OperatorL.compute(LinguisticProcessor.tokenize(result))
st.metric("L du texte généré", f"{final_L['L']:.3f}")
stability_class = f"stability-{final_L['stability']}"
st.markdown(f"<p class='{stability_class}'>État: {final_L['stability']}</p>",
unsafe_allow_html=True)
# TAB 3: Spectral
with tab3:
st.header("🔬 Analyse Spectrale")
if not SpectralAnalysis.is_available():
st.error("❌ Scipy/matplotlib requis")
else:
fragments = st.session_state.shadow_fragments
if not fragments.empty:
word_list = fragments.nlargest(100, 'count')['fragment'].tolist()
selected = st.selectbox("Mot", word_list)
nperseg = st.slider("Fenêtre STFT", 32, 512, 128, 32)
if st.button("📡 Analyser"):
with st.spinner("Analyse..."):
result = SpectralAnalysis.analyze_word(selected, nperseg)
if "error" in result:
st.error(result["error"])
else:
res = result["results"]
fig1, fig2 = result["figures"]
col1, col2, col3 = st.columns(3)
col1.metric("Fréquence", f"{res['freq_dominant']:.4f}")
col2.metric("Omega (ω)", f"{res['omega']:.4f}")
col3.metric("Alpha (α)", f"{res['alpha']:.4f}")
st.pyplot(fig1)
st.pyplot(fig2)
# TAB 4: Évolution λ
with tab4:
st.header("📈 Évolution de λ (Lyapunov)")
lambda_timeline = st.session_state.shadow_cortex.get("lambda_timeline", [])
if lambda_timeline and MATPLOTLIB_AVAILABLE:
recent = lambda_timeline[-1000:]
lambdas = [x['lambda'] for x in recent]
times = list(range(len(lambdas)))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(times, lambdas, 'b-', alpha=0.7, linewidth=1)
ax.axhline(0, color='red', linestyle='--', label='λ = 0 (neutralité)')
ax.axhline(OracleConfig.LEARNING["lambda_target"],
color='green', linestyle='--', label='λ cible')
ax.fill_between(times, -0.05, 0.05, alpha=0.2, color='green')
ax.set_xlabel("Temps (itérations)")
ax.set_ylabel("λ (exposant de Lyapunov)")
ax.set_title("Évolution de la stabilité cognitive")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
# Stats
col1, col2, col3 = st.columns(3)
col1.metric("λ moyen", f"{np.mean(lambdas):.4f}")
col2.metric("λ min", f"{np.min(lambdas):.4f}")
col3.metric("λ max", f"{np.max(lambdas):.4f}")
else:
st.info("Aucune donnée λ disponible. Commencez par apprendre.")
# TAB 5: Auto-Stabilisation
with tab5:
st.header("🎯 Auto-Stabilisation")
st.markdown("""
""")
st.divider()
# Configuration
st.subheader("Paramètres d'auto-stabilisation")
col1, col2 = st.columns(2)
with col1:
target_lambda = st.number_input(
"λ cible",
value=OracleConfig.LEARNING["lambda_target"],
step=0.01,
format="%.4f"
)
OracleConfig.LEARNING["lambda_target"] = target_lambda
with col2:
learning_rate = st.number_input(
"Taux d'apprentissage L",
value=OracleConfig.LEARNING["L_learning_rate"],
step=0.001,
format="%.4f"
)
OracleConfig.LEARNING["L_learning_rate"] = learning_rate
# Test manuel
st.subheader("Test manuel")
test_text = st.text_input("Phrase test")
if st.button("🧪 Tester L"):
if test_text.strip():
tokens = LinguisticProcessor.tokenize(test_text)
result = OperatorL.compute(tokens)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("L", f"{result['L']:.4f}")
col_b.metric("λ", f"{result['lambda']:.4f}")
col_c.metric("Gradient", f"{result['gradient']:.4f}")
col_d.metric("Divergence", f"{result['divergence']:.4f}")
st.write(f"**État**: {result['stability']}")
st.write(f"**Concepts activés**: {', '.join(result['concept_hits'][:10])}")
# Auto-ajustement
if st.button("🔧 Ajuster"):
adjustment = OperatorL.auto_adjust_weights(tokens, target_lambda)
st.success(f"✅ Ajustement: Δ={adjustment['adjustment']:.4f}")
st.write(f"Nouveau λ: {adjustment['new_lambda']:.4f}")
# TAB 6: Export/Import
with tab6:
st.header("⚡ Export / Import")
col1, col2 = st.columns(2)
with col1:
st.subheader("📤 Export")
if st.button("Générer export"):
data = DataExporter.export_all()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
"📥 Télécharger",
data,
file_name=f"oracle_v18_{timestamp}.json",
mime="application/json"
)
with col2:
st.subheader("📥 Import")
import_file = st.file_uploader("Charger export", type=["json"])
if import_file and st.button("Importer"):
json_data = import_file.read().decode('utf-8')
if DataExporter.import_all(json_data):
st.success("✅ Import réussi!")
st.rerun()
# =====================================================
# POINT D'ENTRÉE
# =====================================================
if __name__ == "__main__":
main()
