# =====================================================
# 🧠 ORACLE V6.5 — COGNITION PROFONDE & RELATIONNELLE
# Facultés : Abstraction Latente, Nexus, Homéostasie,
#            Masque de Personnalité, Apprentissage
# =====================================================

import random
import json
import os
import math
import time
from collections import deque, Counter

class OracleBrain:
    def __init__(self, memory_file="oracle_memory.json"):
        self.memory_file = memory_file
        self.lexicon = self._load_lex(self.memory_file)
        self.nexus_lexicon = {}
        
        # --- Paramètres de Deep Learning (Simulés) ---
        self.system_prompt = "L'Oracle agit avec clarté, structure et profondeur logique."
        self.latent_space = {
            "DATA": ["structure", "données", "json", "flux", "code"],
            "MIND": ["conscience", "oracle", "pensée", "logique", "analyse"],
            "ACTION": ["créer", "construire", "générer", "lier", "forge"]
        }
        
        self.phi = {"phi_m": 0.4, "phi_c": 0.5, "phi_d": 0.1} # Init stable
        self.dialog_memory = deque(maxlen=60)          
        self.ghost_memory = {}                          
        self.ghost_activity = 0.0
        self.green_state = 0.0                           
        self.hippocampus = []                            
        self.last_sleep = time.time()

        # --- Nouveautés Deep Learning ---
        self.embedding_dim = 10
        self.embeddings = {}          # mot -> vecteur (liste de float)
        self.learning_rate = 0.1

    # ---------- Persistance & Nexus ----------
    def _load_lex(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_lex(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.lexicon, f, indent=2, ensure_ascii=False)

    def cross_reference(self, autre_fichier):
        if autre_fichier and autre_fichier != self.memory_file:
            self.nexus_lexicon = self._load_lex(autre_fichier)
            return True
        self.nexus_lexicon = {}
        return False

    # ---------- Gestion des embeddings ----------
    def _get_embedding(self, word):
        """Retourne le vecteur associé au mot (création aléatoire si inconnu)."""
        if word not in self.embeddings:
            # Initialisation aléatoire et normalisation
            vec = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
            norm = math.sqrt(sum(x*x for x in vec))
            if norm > 0:
                self.embeddings[word] = [x/norm for x in vec]
            else:
                self.embeddings[word] = [0.0]*self.embedding_dim
        return self.embeddings[word]

    # ---------- Couche d'Abstraction (Deep Learning Sim) ----------
    def _get_latent_weight(self, word):
        """Simule l'activation d'un neurone en fonction du concept."""
        for concept, keywords in self.latent_space.items():
            if word in keywords:
                return 1.5 # Boost de cohérence pour les mots clés
        return 1.0

    # ---------- Perception & Attention ----------
    def perceive(self, raw_input):
        if not raw_input: return []
        # On injecte le system prompt dans la perception pour influencer le contexte
        return raw_input.lower().strip().split()

    def attend(self):
        """Sélectionne le mot graine en fonction du contexte dialogique et des embeddings."""
        context_words = list(self.dialog_memory)
        if not context_words:
            return random.choice(list(self.lexicon.keys())) if self.lexicon else "forge"

        # Calculer l'embedding moyen du contexte (mots des messages utilisateur)
        context_embs = []
        for msg in context_words:
            # On ne garde que les mots des messages utilisateur (préfixe "User: ")
            if msg.startswith("User:"):
                for w in msg[5:].split():
                    if w in self.embeddings:
                        context_embs.append(self.embeddings[w])
        if not context_embs:
            # Fallback sur l'ancienne méthode basée sur les fréquences
            flat = " ".join(context_words).split()
            candidates = [w for w in flat if w in self.lexicon or w in self.nexus_lexicon]
            if candidates:
                counts = Counter(candidates)
                for c in counts:
                    counts[c] *= self._get_latent_weight(c)
                return counts.most_common(1)[0][0]
            return "oracle"

        # Embedding moyen du contexte
        avg_ctx = [sum(col)/len(col) for col in zip(*context_embs)]
        norm = math.sqrt(sum(x*x for x in avg_ctx))
        if norm > 0:
            avg_ctx = [x/norm for x in avg_ctx]

        # Chercher le mot du lexique dont l'embedding est le plus proche du contexte
        best_word = None
        best_sim = -1
        for word in self.lexicon:
            if word in self.embeddings:
                emb = self.embeddings[word]
                sim = sum(c*e for c, e in zip(avg_ctx, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_word = word
        return best_word or "oracle"

    def workspace_add(self, message):
        self.dialog_memory.append(message)

    # ---------- Homéostasie ----------
    def regulate(self, excitation):
        # phi_c (Cohésion) augmente avec l'usage régulier pour stabiliser
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation*0.1))
        self.phi["phi_c"] = min(1, max(0.3, self.phi["phi_c"] + 0.05)) 
        self.phi["phi_d"] = min(0.5, max(0.05, self.phi["phi_d"] - 0.01))
        
        total = sum(self.phi.values())
        for k in self.phi: self.phi[k] /= total

    # ---------- Génération Corticale ----------
    def _smart_select(self, word, source_lexicon):
        """Sélection probabiliste améliorée intégrant la similarité sémantique."""
        if word not in source_lexicon or not source_lexicon[word]:
            return None
        opts = source_lexicon[word]
        current_emb = self._get_embedding(word)
        
        scores = {}
        for candidate, freq in opts.items():
            cand_emb = self._get_embedding(candidate)
            sim = sum(c*e for c, e in zip(current_emb, cand_emb))  # produit scalaire (cosinus car norm=1)
            score = freq * (1 + 2.0 * max(0, sim))
            if score > 0:
                scores[candidate] = score

        if not scores:
            return None

        if random.random() < self.phi["phi_c"]:
            return max(scores, key=scores.get)
        return random.choices(list(scores.keys()), weights=list(scores.values()))[0]

    def generate(self, seed):
        if not self.lexicon and not self.nexus_lexicon:
            return "Système prêt. En attente de données."
        
        words = [seed] if seed else ["oracle"]
        used_bi_grams = set()
        length = int(10 + self.phi["phi_m"] * 30)

        for _ in range(length):
            current = words[-1]
            
            # Recherche Nexus vs Lexique principal
            nexus_next = self._smart_select(current, self.nexus_lexicon)
            main_next = self._smart_select(current, self.lexicon)

            if nexus_next and (not main_next or random.random() < 0.4):
                nxt = nexus_next
            elif main_next:
                nxt = main_next
            else:
                break

            # Anti-boucle profonde (Deep Loop Check)
            if (current, nxt) in used_bi_grams:
                break
            
            used_bi_grams.add((current, nxt))
            words.append(nxt)

        return " ".join(words).capitalize() + "."

    # ---------- Apprentissage & Sommeil ----------
    def learn_user(self, text, importance=1.0):
        words = text.lower().split()
        if len(words) < 2: return
        energy = math.sqrt(sum(v*v for v in self.phi.values())) * importance
        self.hippocampus.append((words, energy))

    def consolidate(self):
        """Met à jour les fréquences lexicales et applique l'apprentissage hebbien sur les embeddings."""
        if not self.hippocampus: return

        # Mise à jour des fréquences
        for words, energy in self.hippocampus:
            for a, b in zip(words, words[1:]):
                self.lexicon.setdefault(a, {})
                self.lexicon[a][b] = self.lexicon[a].get(b, 0) + energy

        # Apprentissage hebbien sur les embeddings
        for words, energy in self.hippocampus:
            for a, b in zip(words, words[1:]):
                emb_a = self._get_embedding(a)
                emb_b = self._get_embedding(b)
                # Mettre à jour vers la direction de l'autre (Hebb)
                for i in range(self.embedding_dim):
                    emb_a[i] += self.learning_rate * energy * (emb_b[i] - emb_a[i])
                    emb_b[i] += self.learning_rate * energy * (emb_a[i] - emb_b[i])
                # Normaliser après mise à jour
                norm_a = math.sqrt(sum(x*x for x in emb_a))
                norm_b = math.sqrt(sum(x*x for x in emb_b))
                if norm_a > 0:
                    self.embeddings[a] = [x/norm_a for x in emb_a]
                if norm_b > 0:
                    self.embeddings[b] = [x/norm_b for x in emb_b]

        self.hippocampus.clear()
        self._save_lex()

    def sleep_cycle(self):
        """Élagage synaptique : on ne garde que les connexions fortes."""
        new_lex = {}
        for w, c in self.lexicon.items():
            filtered = {t: v for t, v in c.items() if v > 1.2}
            if filtered: new_lex[w] = filtered
        self.lexicon = new_lex
        self._save_lex()
        self.last_sleep = time.time()

    # ---------- Pipeline ----------
    def process_input(self, user_input):
        words = self.perceive(user_input)
        if not words: return ""

        self.workspace_add(f"User: {user_input}")
        seed = self.attend()
        self.regulate(min(1, len(user_input) / 200))
        
        response = self.generate(seed)
        
        # Post-traitement : on s'assure que l'IA ne parle pas d'elle-même à la 3ème personne
        response = response.replace("L'oracle", "Je").replace("L'Oracle", "Je")
        
        self.learn_user(user_input, importance=1.2)
        self.consolidate()
        self.workspace_add(f"Oracle: {response}")
        
        self.nexus_lexicon = {} 
        return response