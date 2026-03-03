# =====================================================
# 🧠 ORACLE V6.5 — CŒUR COGNITIF PROFOND
# Facultés : Abstraction Latente (embeddings),
#            Attention contextuelle, Nexus,
#            Homéostasie, Masque, Apprentissage hebbien
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

        # --- Paramètres du Masque (personnalité) ---
        self.system_prompt = "L'Oracle agit avec clarté, structure et profondeur logique."
        self.latent_space = {
            "DATA": ["structure", "données", "json", "flux", "code"],
            "MIND": ["conscience", "oracle", "pensée", "logique", "analyse"],
            "ACTION": ["créer", "construire", "générer", "lier", "forge"]
        }

        # --- Homéostasie ---
        self.phi = {"phi_m": 0.4, "phi_c": 0.5, "phi_d": 0.1}
        self.dialog_memory = deque(maxlen=60)
        self.ghost_memory = {}
        self.ghost_activity = 0.0
        self.green_state = 0.0
        self.hippocampus = []
        self.last_sleep = time.time()

        # --- Deep Learning (simulé) ---
        self.embedding_dim = 10
        self.embeddings = {}          # mot -> vecteur normalisé
        self.learning_rate = 0.1

        # --- Connecteurs logiques (injecteur) ---
        self.connecteurs = [
            "cependant", "néanmoins", "d'autre part", "en revanche",
            "par ailleurs", "toutefois", "ainsi", "donc", "par conséquent"
        ]

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
            vec = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
            norm = math.sqrt(sum(x*x for x in vec))
            if norm > 0:
                self.embeddings[word] = [x/norm for x in vec]
            else:
                self.embeddings[word] = [0.0]*self.embedding_dim
        return self.embeddings[word]

    # ---------- Perception & Attention ----------
    def perceive(self, raw_input):
        if not raw_input:
            return []
        return raw_input.lower().strip().split()

    def attend(self):
        """Sélectionne le mot graine en fonction du contexte dialogique et des embeddings."""
        context_words = list(self.dialog_memory)
        if not context_words:
            return random.choice(list(self.lexicon.keys())) if self.lexicon else "forge"

        # Embedding moyen du contexte (uniquement messages utilisateur)
        context_embs = []
        for msg in context_words:
            if msg.startswith("User:"):
                for w in msg[5:].split():
                    if w in self.embeddings:
                        context_embs.append(self.embeddings[w])
        if not context_embs:
            # Fallback fréquentiel
            flat = " ".join(context_words).split()
            candidates = [w for w in flat if w in self.lexicon or w in self.nexus_lexicon]
            if candidates:
                counts = Counter(candidates)
                return counts.most_common(1)[0][0]
            return "oracle"

        avg_ctx = [sum(col)/len(col) for col in zip(*context_embs)]
        norm = math.sqrt(sum(x*x for x in avg_ctx))
        if norm > 0:
            avg_ctx = [x/norm for x in avg_ctx]

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
    def _green_noise(self):
        self.green_state = 0.92 * self.green_state + 0.08 * random.uniform(-1, 1)
        return abs(self.green_state) < 0.25

    def regulate(self, excitation):
        self.phi["phi_m"] = min(1, max(0.1, self.phi["phi_m"] + excitation*0.1))
        self.phi["phi_c"] = min(1, max(0.3, self.phi["phi_c"] + 0.05))
        self.phi["phi_d"] = min(0.5, max(0.05, self.phi["phi_d"] - 0.01))
        total = sum(self.phi.values())
        for k in self.phi:
            self.phi[k] /= total
        self._green_noise()

    # ---------- Génération Corticale (avec scores combinés) ----------
    def _compute_score(self, word, candidate, freq, source):
        """Calcule le score d'un candidat : fréquence * (1 + similarité + boost masque)."""
        current_emb = self._get_embedding(word)
        cand_emb = self._get_embedding(candidate)
        sim = sum(c*e for c, e in zip(current_emb, cand_emb))  # cosinus

        # Boost du masque : si le candidat appartient au même concept latent que le contexte
        boost = 1.0
        for concept, mots in self.latent_space.items():
            if candidate in mots:
                boost = 1.5
                break

        return freq * (1 + 2.0 * max(0, sim)) * boost

    def _smart_select(self, word, source_lexicon):
        """Sélection probabiliste avec scores mixtes (fréquence, sémantique, masque)."""
        if word not in source_lexicon or not source_lexicon[word]:
            return None
        opts = source_lexicon[word]

        scores = {}
        for candidate, freq in opts.items():
            score = self._compute_score(word, candidate, freq, source_lexicon)
            if score > 0:
                scores[candidate] = score

        if not scores:
            return None

        if random.random() < self.phi["phi_c"]:
            return max(scores, key=scores.get)
        return random.choices(list(scores.keys()), weights=list(scores.values()))[0]

    def _inject_connecteurs(self, words):
        """Ajoute un connecteur logique si la phrase est trop courte ou sans liaison."""
        # Si phrase très courte (< 5 mots), ajouter un connecteur au début
        if len(words) < 5:
            connecteur = random.choice(self.connecteurs)
            words.insert(0, connecteur)
        # Sinon, si aucun connecteur n'est présent, en insérer un au milieu
        elif not any(w in self.connecteurs for w in words):
            pos = random.randint(1, len(words)-1)
            words.insert(pos, random.choice(self.connecteurs))
        return words

    def generate(self, seed):
        if not self.lexicon and not self.nexus_lexicon:
            return "Système prêt. En attente de données."

        words = [seed] if seed else ["oracle"]
        used_bi_grams = set()
        length = int(10 + self.phi["phi_m"] * 30)

        for _ in range(length):
            current = words[-1]

            # Cherche d'abord dans le Nexus, puis dans le lexique principal
            nexus_next = self._smart_select(current, self.nexus_lexicon)
            main_next = self._smart_select(current, self.lexicon)

            if nexus_next and (not main_next or random.random() < 0.4):
                nxt = nexus_next
                self.ghost_activity = 0.7 * self.ghost_activity + 0.3
            elif main_next:
                nxt = main_next
            else:
                break

            if (current, nxt) in used_bi_grams:
                break
            used_bi_grams.add((current, nxt))
            words.append(nxt)

        # Injection de connecteurs si nécessaire
        words = self._inject_connecteurs(words)
        return " ".join(words).capitalize() + "."

    # ---------- Apprentissage & Sommeil ----------
    def learn_user(self, text, importance=1.0):
        words = text.lower().split()
        if len(words) < 2:
            return
        energy = math.sqrt(sum(v*v for v in self.phi.values())) * importance
        self.hippocampus.append((words, energy))

    def consolidate(self):
        """Met à jour les fréquences lexicales et applique l'apprentissage hebbien."""
        if not self.hippocampus or not self._green_noise():
            return

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
                for i in range(self.embedding_dim):
                    emb_a[i] += self.learning_rate * energy * (emb_b[i] - emb_a[i])
                    emb_b[i] += self.learning_rate * energy * (emb_a[i] - emb_b[i])
                norm_a = math.sqrt(sum(x*x for x in emb_a))
                norm_b = math.sqrt(sum(x*x for x in emb_b))
                if norm_a > 0:
                    self.embeddings[a] = [x/norm_a for x in emb_a]
                if norm_b > 0:
                    self.embeddings[b] = [x/norm_b for x in emb_b]

        self.hippocampus.clear()
        self._save_lex()

    def sleep_cycle(self):
        """Élagage synaptique : ne garde que les connexions fortes."""
        new_lex = {}
        for w, c in self.lexicon.items():
            filtered = {t: v for t, v in c.items() if v > 1.2}
            if filtered:
                new_lex[w] = filtered
        self.lexicon = new_lex
        self._save_lex()
        self.last_sleep = time.time()

    # ---------- Pipeline ----------
    def process_input(self, user_input):
        words = self.perceive(user_input)
        if not words:
            return ""

        self.workspace_add(f"User: {user_input}")
        seed = self.attend()
        self.regulate(min(1, len(user_input) / 200))

        response = self.generate(seed)

        # Post-traitement : éviter l'auto-référence à la 3ème personne
        response = response.replace("L'oracle", "Je").replace("L'Oracle", "Je")

        self.learn_user(user_input, importance=1.2)
        self.consolidate()
        self.workspace_add(f"Oracle: {response}")

        self.nexus_lexicon = {}  # Réinitialisation du Nexus après usage
        return response