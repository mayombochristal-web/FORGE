import sqlite3
import os
import re
from collections import Counter

# =========================================================
# 1. CONFIGURATION
# =========================================================
DB_FOLDER = "oracle_memory"
DB_PATH = os.path.join(DB_FOLDER, "relations.db")
CORPUS_SOURCE = "corpus.txt"

os.makedirs(DB_FOLDER, exist_ok=True)

# =========================================================
# 2. CORPUS INTÉGRÉ
# =========================================================
BIG_CORPUS = """
L'intelligence artificielle est une discipline scientifique qui explore les capacités des machines à simuler l'intelligence humaine.
Au cœur de cette technologie, le modèle Transformer utilise l'attention pour comprendre le langage.
Le savoir est un réseau de relations dynamiques entre les concepts.
Chaque mot traité par l'algorithme devient un jeton numérique dans une base de données SQLite.
La logique mathématique rencontre la créativité dans le développement des réseaux de neurones profonds.
L'apprentissage supervisé permet de prédire le mot suivant avec une grande précision.
L'éthique et la sécurité sont fondamentales pour construire une IA bénéfique à l'humanité.
Les processeurs modernes permettent d'entraîner des modèles complexes sur des millions de paramètres.
La singularité technologique représente le point où l'IA surpasse les capacités d'analyse humaine.
Le traitement du langage naturel transforme notre interaction avec les systèmes informatiques.
"""

# =========================================================
# 3. INITIALISATION DB AMÉLIORÉE
# =========================================================
def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Bigram table
    c.execute('''
        CREATE TABLE IF NOT EXISTS bigrams (
            word TEXT,
            next_word TEXT,
            frequency INTEGER,
            probability REAL,
            PRIMARY KEY (word, next_word)
        )
    ''')

    # Index pour accélérer
    c.execute('CREATE INDEX IF NOT EXISTS idx_word ON bigrams(word)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_next_word ON bigrams(next_word)')

    conn.commit()
    conn.close()
    print("📁 Base SQLite optimisée initialisée.")

# =========================================================
# 4. NETTOYAGE TEXTE
# =========================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûù\d\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()

# =========================================================
# 5. INGESTION OPTIMISÉE
# =========================================================
def ingest_data(text):
    words = clean_text(text)

    if len(words) < 2:
        print("⚠️ Corpus insuffisant.")
        return

    print(f"🧠 Traitement de {len(words)} tokens...")

    # Comptage rapide en mémoire
    bigram_counts = Counter(zip(words[:-1], words[1:]))

    # Calcul total par mot
    totals = Counter(words[:-1])

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for (word, next_word), freq in bigram_counts.items():
        prob = freq / totals[word]

        c.execute('''
            INSERT INTO bigrams (word, next_word, frequency, probability)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(word, next_word)
            DO UPDATE SET
                frequency = frequency + ?,
                probability = ?
        ''', (word, next_word, freq, prob, freq, prob))

    conn.commit()
    conn.close()
    print("✅ Ingestion probabiliste terminée.")

# =========================================================
# 6. GÉNÉRATION SIMPLE PROBABILISTE
# =========================================================
def generate_text(seed, max_length=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    current_word = seed.lower()
    result = [current_word]

    for _ in range(max_length):
        c.execute('''
            SELECT next_word, probability
            FROM bigrams
            WHERE word = ?
        ''', (current_word,))

        rows = c.fetchall()
        if not rows:
            break

        words, probs = zip(*rows)

        import random
        next_word = random.choices(words, weights=probs)[0]

        result.append(next_word)
        current_word = next_word

    conn.close()
    return " ".join(result)

# =========================================================
# 7. EXECUTION
# =========================================================
if __name__ == "__main__":
    initialize_db()

    if not os.path.exists(CORPUS_SOURCE):
        with open(CORPUS_SOURCE, "w", encoding="utf-8") as f:
            f.write(BIG_CORPUS)

    with open(CORPUS_SOURCE, "r", encoding="utf-8") as f:
        data = f.read()

    ingest_data(data)

    print("\n🔮 Exemple génération :")
    print(generate_text("intelligence"))