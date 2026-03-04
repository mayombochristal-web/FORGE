import sqlite3
import os
import re

# =========================================================
# CONFIGURATION
# =========================================================
DB_FOLDER = "oracle_memory"
DB_PATH = os.path.join(DB_FOLDER, "relations.db")
CORPUS_SOURCE = "corpus.txt"  # Placez vos textes dans ce fichier

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

def initialize_db():
    """Crée la structure de la base de données"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table ngrams : stocke les transitions entre les mots
    c.execute('''CREATE TABLE IF NOT EXISTS ngrams
                 (word TEXT, next_word TEXT, frequency INTEGER,
                  PRIMARY KEY (word, next_word))''')
    conn.commit()
    conn.close()

def clean_text(text):
    """Nettoyage simple pour le tokenizer dynamique"""
    text = text.lower()
    # On garde les lettres, les chiffres et une ponctuation basique
    text = re.sub(r"[^a-zàâçéèêëîïôûù\d\s']", " ", text)
    return text.split()

def ingest_text(file_path):
    """Analyse le texte et remplit la base de données"""
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Créez d'abord un fichier '{file_path}' avec vos textes.")
        return

    print(f"📖 Lecture de {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    words = clean_text(content)
    
    if len(words) < 2:
        print("⚠️ Pas assez de mots pour créer des relations.")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("🧠 Analyse des relations (N-grams)...")
    # On crée des paires (mot_actuel, mot_suivant)
    for i in range(len(words) - 1):
        word = words[i]
        next_word = words[i+1]

        # Mise à jour de la fréquence en cas de répétition
        c.execute('''INSERT INTO ngrams (word, next_word, frequency)
                     VALUES (?, ?, 1)
                     ON CONFLICT(word, next_word) 
                     DO UPDATE SET frequency = frequency + 1''', (word, next_word))

    conn.commit()
    conn.close()
    print(f"✅ Ingestion terminée. Base '{DB_PATH}' prête.")

# =========================================================
# EXÉCUTION
# =========================================================
if __name__ == "__main__":
    initialize_db()
    
    # Création d'un fichier exemple si vide pour le test
    if not os.path.exists(CORPUS_SOURCE):
        with open(CORPUS_SOURCE, "w", encoding="utf-8") as f:
            f.write("Dans le texte fondateur de l'intelligence artificielle, "
                    "les modèles de langage apprennent des relations entre les mots.")
    
    ingest_text(CORPUS_SOURCE)
