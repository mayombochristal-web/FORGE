import pandas as pd
import numpy as np
import json, os, re, datetime, feedparser
from collections import Counter

# --- CONFIGURATION DES CHEMINS (Doit pointer vers votre dossier V12) ---
MEM = "oracle_memory"
FILES = {
    "fragments": f"{MEM}/fragments.csv",
    "relations": f"{MEM}/relations.json",
    "cortex": f"{MEM}/cortex.json"
}

# --- SOURCES D'APPRENTISSAGE (Flux RSS Scientifiques/Techniques) ---
SOURCES = [
    "https://arxiv.org/rss/math.DS",      # Systèmes Dynamiques
    "https://arxiv.org/rss/cs.AI",       # Intelligence Artificielle
    "https://feeds.feedburner.com/sciencedaily/computers_math/mathematics"
]

# --- CHARGEMENT DES DONNÉES ---
def load_memory():
    if not os.path.exists(FILES["cortex"]):
        return None, None, None
    frag = pd.read_csv(FILES["fragments"])
    with open(FILES["relations"], "r") as f:
        rel = json.load(f)
    with open(FILES["cortex"], "r") as f:
        ctx = json.load(f)
    return frag, rel, ctx

# --- LOGIQUE DE NETTOYAGE ---
def clean_text(t):
    return re.sub(r"[^a-zàâéèêëîïôùûüœ\s]", " ", t.lower())

def tokenize(t):
    return [w for w in clean_text(t).split() if len(w) > 1]

# --- FILTRE DE STABILITÉ (LYAPUNOV) ---
def check_stability(new_text, current_is):
    """
    Vérifie si le texte entrant ne va pas faire chuter l'Indice de Sincérité (IS).
    """
    words = tokenize(new_text)
    if not words: return False
    
    # Calcul de l'énergie prédictive
    # Plus les mots sont nouveaux ou rares, plus l'énergie est élevée (proche de 1)
    # Si le texte est répétitif ou vide de sens, l'énergie chute.
    simulated_energy = 1 / (1 + (len(words) / 100)) # Approximation rapide
    
    # Seuil de Lyapunov : on rejette si l'énergie est trop faible (bruit blanc)
    if simulated_energy < 0.10: 
        return False
    return True

# --- FONCTION D'APPRENTISSAGE AUTONOME ---
def worker_learn(text, frag, rel, ctx):
    words = tokenize(text)
    if not words: return
    
    # Mise à jour fragments
    counts = Counter(words)
    for w, c in counts.items():
        mask = frag["fragment"] == w
        if mask.any():
            frag.loc[mask, "count"] += c
        else:
            new_row = pd.DataFrame([{"fragment": w, "count": c}])
            frag = pd.concat([frag, new_row], ignore_index=True)
            
    # Mise à jour relations (N-Grams n=2 pour le worker)
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        rel.setdefault(a, {})
        rel[a][b] = rel[a].get(b, 0) + 1
        
    # Mise à jour Cortex et Timeline
    ctx["age"] += len(words)
    ctx["VS"] = 10 + np.log1p(ctx["age"])
    
    for w in words:
        ctx["timeline"].append({"w": w, "e": 1/(1 + len(words))})
    
    # Limitation de la taille de la timeline pour éviter la saturation mémoire
    if len(ctx["timeline"]) > 5000:
        ctx["timeline"] = ctx["timeline"][-5000:]
        
    return frag, rel, ctx

# --- CYCLE PRINCIPAL DU WORKER ---
def run_worker():
    print(f"[{datetime.datetime.now()}] 🧠 Oracle Worker : Début du cycle...")
    frag, rel, ctx = load_memory()
    
    if frag is None:
        print("❌ Erreur : Mémoire introuvable. Lancez l'application Streamlit d'abord.")
        return

    total_new_words = 0
    
    for url in SOURCES:
        print(f"🌐 Scraping : {url}")
        feed = feedparser.parse(url)
        
        for entry in feed.entries[:3]: # 3 articles par source pour éviter la surcharge
            raw_content = f"{entry.title} {entry.description}"
            
            # Application du filtre de stabilité
            if check_stability(raw_content, ctx.get("IS", 1.0)):
                frag, rel, ctx = worker_learn(raw_content, frag, rel, ctx)
                total_new_words += len(tokenize(raw_content))
                print(f" ✅ Assimilé : {entry.title[:40]}...")
            else:
                print(f" ⚠️ Rejeté (Chaos détecté) : {entry.title[:40]}...")

    # Sauvegarde des résultats
    frag.to_csv(FILES["fragments"], index=False)
    with open(FILES["relations"], "w") as f: json.dump(rel, f)
    with open(FILES["cortex"], "w") as f: json.dump(ctx, f)
    
    print(f"🏁 Cycle terminé. {total_new_words} mots ajoutés à l'Ancrage.")

if __name__ == "__main__":
    run_worker()
