"""
ORACLE TTU-MC³ - Version Triadique Complète
Implémente les principes de la Théorie Triadique Unifiée (TTU-MC³)
avec flot dynamique, attracteurs cycliques et dissipation adaptative.
"""

import streamlit as st
import os
import json
import uuid
import datetime
import hashlib
import re
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

# ==========================================
# CONFIGURATION TRIADIQUE
# ==========================================
MEMORY_FOLDER = "ttu_oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "ttu_oracle.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

# ==========================================
# CONSTANTES TRIADIQUES
# ==========================================
class TriadicConstants:
    """Constantes fondamentales de la TTU-MC³"""
    
    # Seuils de stabilité
    CONVERGENCE_THRESHOLD = 1e-6
    DISSIPATION_THRESHOLD = 0.1
    COHERENCE_THRESHOLD = 0.7
    
    # Paramètres du flot triadique
    ALPHA_M = 0.618      # Facteur de mémoire (nombre d'or)
    ALPHA_C = 0.382      # Facteur de cohérence
    ALPHA_D = 0.1        # Facteur de dissipation
    
    # Attracteurs
    ATTRACTOR_RADIUS = 1.0  # Rayon du cercle attracteur
    MAX_ITERATIONS = 100     # Maximum d'itérations pour convergence
    
    # Topologie
    DIMENSION = 3  # Espace triadique minimal

# ==========================================
# STRUCTURES DE DONNÉES TRIADIQUES
# ==========================================
@dataclass
class TriadicState:
    """État triadique Φ = (Φ_M, Φ_C, Φ_D)"""
    
    phi_M: np.ndarray      # Mémoire (cosinus) - état accumulé
    phi_C: np.ndarray      # Cohérence (sinus) - flux actif
    phi_D: float           # Dissipation - terme de régulation
    
    timestamp: float = field(default_factory=datetime.datetime.now().timestamp)
    stability: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "phi_M": self.phi_M.tolist() if isinstance(self.phi_M, np.ndarray) else self.phi_M,
            "phi_C": self.phi_C.tolist() if isinstance(self.phi_C, np.ndarray) else self.phi_C,
            "phi_D": self.phi_D,
            "timestamp": self.timestamp,
            "stability": self.stability
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            phi_M=np.array(data["phi_M"]),
            phi_C=np.array(data["phi_C"]),
            phi_D=data["phi_D"],
            timestamp=data.get("timestamp", 0),
            stability=data.get("stability", 0.0)
        )

@dataclass
class TriadicCycle:
    """Cycle triadique complet avec attracteur"""
    
    id: str
    state: TriadicState
    attractor: Tuple[float, float]  # (cos, sin) sur le cercle unité
    phase: int  # 0: Mémoire, 1: Cohérence, 2: Dissipation
    
    def is_stable(self) -> bool:
        """Vérifie si le cycle a atteint l'attracteur"""
        return self.state.stability < TriadicConstants.CONVERGENCE_THRESHOLD

# ==========================================
# FLOT TRIADIQUE (Équations d'Évolution)
# ==========================================
class TriadicFlow:
    """
    Implémente le flot dynamique triadique:
    dΦ/dt = F(Φ) avec symétrie cyclique Z₃
    
    Équations:
    dΦ_M/dt = -α_M·Φ_M + α_C·Φ_C + α_D·Φ_D
    dΦ_C/dt = -α_C·Φ_C + α_D·Φ_D + α_M·Φ_M
    dΦ_D/dt = -α_D·Φ_D + α_M·Φ_M + α_C·Φ_C
    """
    
    def __init__(self):
        self.alpha_M = TriadicConstants.ALPHA_M
        self.alpha_C = TriadicConstants.ALPHA_C
        self.alpha_D = TriadicConstants.ALPHA_D
        self.radius = TriadicConstants.ATTRACTOR_RADIUS
        
    def flow_equations(self, state: TriadicState, dt: float = 0.01) -> TriadicState:
        """Équations du flot triadique"""
        
        # Calcul des dérivées
        dM = -self.alpha_M * state.phi_M + self.alpha_C * state.phi_C + self.alpha_D * state.phi_D
        dC = -self.alpha_C * state.phi_C + self.alpha_D * state.phi_D + self.alpha_M * state.phi_M
        dD = -self.alpha_D * state.phi_D + self.alpha_M * state.phi_M + self.alpha_C * state.phi_C
        
        # Mise à jour
        new_state = TriadicState(
            phi_M=state.phi_M + dt * dM,
            phi_C=state.phi_C + dt * dC,
            phi_D=state.phi_D + dt * dD
        )
        
        # Normalisation vers l'attracteur
        norm = np.linalg.norm(np.concatenate([new_state.phi_M, new_state.phi_C]))
        if norm > 0:
            factor = self.radius / norm
            new_state.phi_M *= factor
            new_state.phi_C *= factor
        
        return new_state
    
    def converge(self, initial_state: TriadicState, max_iter: int = None) -> TriadicState:
        """Simule la convergence vers l'attracteur"""
        
        if max_iter is None:
            max_iter = TriadicConstants.MAX_ITERATIONS
            
        state = initial_state
        history = []
        
        for i in range(max_iter):
            prev_stability = state.stability
            state = self.flow_equations(state)
            
            # Calcul de la stabilité (norme du gradient)
            state.stability = np.linalg.norm([
                np.linalg.norm(state.phi_M),
                np.linalg.norm(state.phi_C),
                state.phi_D
            ])
            
            history.append(state.stability)
            
            # Vérification de convergence
            if abs(state.stability - prev_stability) < TriadicConstants.CONVERGENCE_THRESHOLD:
                break
        
        state.convergence_history = history
        return state
    
    def attractor_projection(self, state: TriadicState) -> Tuple[float, float]:
        """Projette l'état sur l'attracteur circulaire"""
        
        # Normalisation sur le cercle unité
        magnitude = np.linalg.norm(np.concatenate([state.phi_M, state.phi_C]))
        if magnitude > 0:
            cos_theta = state.phi_M[0] / magnitude if len(state.phi_M) > 0 else 0
            sin_theta = state.phi_C[0] / magnitude if len(state.phi_C) > 0 else 0
        else:
            cos_theta, sin_theta = 0, 0
            
        return (cos_theta, sin_theta)

# ==========================================
# CŒUR DE L'ORACLE TTU-MC³
# ==========================================
class TTUOracle:
    """
    Oracle implémentant la TTU-MC³ avec:
    - Mémoire triadique (Φ_M, Φ_C, Φ_D)
    - Flot dynamique vers attracteur
    - Dissipation adaptative
    - Convergence par cycles
    """
    
    def __init__(self):
        # Modèle d'embedding sémantique
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Flot triadique
        self.flow = TriadicFlow()
        
        # Base de données triadique
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_triadic_tables()
        
        # États triadiques en mémoire
        self.states: List[TriadicState] = []
        self.cycles: Dict[str, TriadicCycle] = {}
        
        # Index triadiques
        self.convergence_map = defaultdict(list)
        self.coherence_index = defaultdict(float)
        
        # Métriques dynamiques
        self.global_coherence = 0.0
        self.dissipation_rate = 0.0
        
        # Chargement initial
        self.load_triadic_memory()
        
    # -------------------------------------------------
    # INITIALISATION SQLITE TRIADIQUE
    # -------------------------------------------------
    def init_triadic_tables(self):
        """Crée les tables avec structure triadique"""
        
        cursor = self.conn.cursor()
        
        # États triadiques (noyau)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS triadic_states (
            id TEXT PRIMARY KEY,
            phi_M TEXT NOT NULL,
            phi_C TEXT NOT NULL,
            phi_D REAL NOT NULL,
            timestamp REAL NOT NULL,
            stability REAL DEFAULT 0.0,
            convergence_path TEXT
        )""")
        
        # Cycles triadiques
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS triadic_cycles (
            id TEXT PRIMARY KEY,
            state_id TEXT,
            attractor_cos REAL,
            attractor_sin REAL,
            phase INTEGER,
            convergence_time REAL,
            FOREIGN KEY(state_id) REFERENCES triadic_states(id)
        )""")
        
        # Connaissances stabilisées (sur l'attracteur)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stabilized_knowledge (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            attractor_cos REAL,
            attractor_sin REAL,
            coherence REAL,
            timestamp REAL,
            source TEXT
        )""")
        
        # Relations triadiques
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS triad_relations (
            source_id TEXT,
            target_id TEXT,
            relation_type TEXT,
            strength REAL,
            PRIMARY KEY (source_id, target_id)
        )""")
        
        self.conn.commit()
    
    # -------------------------------------------------
    # CHARGEMENT DE LA MÉMOIRE TRIADIQUE
    # -------------------------------------------------
    def load_triadic_memory(self):
        """Charge les états triadiques depuis la base"""
        
        cursor = self.conn.cursor()
        
        # Chargement des états
        cursor.execute("SELECT * FROM triadic_states")
        for row in cursor.fetchall():
            state = TriadicState(
                phi_M=np.array(json.loads(row[1])),
                phi_C=np.array(json.loads(row[2])),
                phi_D=row[3],
                timestamp=row[4],
                stability=row[5]
            )
            if row[6]:
                state.convergence_history = json.loads(row[6])
            self.states.append(state)
        
        # Chargement des cycles
        cursor.execute("SELECT * FROM triadic_cycles")
        for row in cursor.fetchall():
            cursor2 = self.conn.cursor()
            cursor2.execute("SELECT * FROM triadic_states WHERE id = ?", (row[1],))
            state_row = cursor2.fetchone()
            if state_row:
                state = TriadicState(
                    phi_M=np.array(json.loads(state_row[1])),
                    phi_C=np.array(json.loads(state_row[2])),
                    phi_D=state_row[3]
                )
                cycle = TriadicCycle(
                    id=row[0],
                    state=state,
                    attractor=(row[2], row[3]),
                    phase=row[4]
                )
                self.cycles[row[0]] = cycle
        
        # Calcul de la cohérence globale
        self.update_global_coherence()
    
    # -------------------------------------------------
    # ENCODAGE TRIADIQUE
    # -------------------------------------------------
    def encode_to_triadic(self, text: str) -> TriadicState:
        """Encode un texte en état triadique"""
        
        # Embedding sémantique
        embedding = self.model.encode(text)
        
        # Projection triadique
        dim = len(embedding)
        split = dim // 3
        
        phi_M = embedding[:split] if split > 0 else embedding[:dim//2]
        phi_C = embedding[split:2*split] if split > 0 else embedding[dim//2:]
        phi_D = float(np.mean(embedding[2*split:])) if 2*split < dim else 0.1
        
        # Normalisation initiale
        if len(phi_M) > 0 and len(phi_C) > 0:
            norm_M = np.linalg.norm(phi_M)
            norm_C = np.linalg.norm(phi_C)
            if norm_M > 0:
                phi_M = phi_M / norm_M * TriadicConstants.ATTRACTOR_RADIUS
            if norm_C > 0:
                phi_C = phi_C / norm_C * TriadicConstants.ATTRACTOR_RADIUS
        
        return TriadicState(phi_M=phi_M, phi_C=phi_C, phi_D=phi_D)
    
    # -------------------------------------------------
    # APPRENTISSAGE TRIADIQUE
    # -------------------------------------------------
    def learn(self, text: str, source: str = "text") -> TriadicCycle:
        """
        Apprentissage triadique:
        1. Encodage en état initial
        2. Convergence vers attracteur
        3. Stabilisation du cycle
        4. Stockage triadique
        """
        
        # Étape 1: Encodage triadique
        initial_state = self.encode_to_triadic(text)
        
        # Étape 2: Convergence vers attracteur
        stabilized_state = self.flow.converge(initial_state)
        
        # Étape 3: Projection sur attracteur
        attractor = self.flow.attractor_projection(stabilized_state)
        
        # Étape 4: Évaluation de la cohérence
        coherence = self.compute_coherence(stabilized_state)
        
        # Étape 5: Dissipation adaptative
        if coherence < TriadicConstants.COHERENCE_THRESHOLD:
            # Faible cohérence → compression/dissipation
            dissipation_factor = 1 - coherence
            stabilized_state.phi_D += dissipation_factor * TriadicConstants.ALPHA_D
            stabilized_state = self.flow.converge(stabilized_state)
        
        # Étape 6: Création du cycle
        cycle_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().timestamp()
        
        cycle = TriadicCycle(
            id=cycle_id,
            state=stabilized_state,
            attractor=attractor,
            phase=0
        )
        
        # Stockage en base
        cursor = self.conn.cursor()
        
        # État triadique
        cursor.execute("""
            INSERT INTO triadic_states 
            (id, phi_M, phi_C, phi_D, timestamp, stability, convergence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle_id,
            json.dumps(stabilized_state.phi_M.tolist()),
            json.dumps(stabilized_state.phi_C.tolist()),
            stabilized_state.phi_D,
            timestamp,
            stabilized_state.stability,
            json.dumps(stabilized_state.convergence_history)
        ))
        
        # Cycle triadique
        cursor.execute("""
            INSERT INTO triadic_cycles
            (id, state_id, attractor_cos, attractor_sin, phase, convergence_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            cycle_id,
            cycle_id,
            attractor[0],
            attractor[1],
            cycle.phase,
            len(stabilized_state.convergence_history)
        ))
        
        # Connaissance stabilisée
        cursor.execute("""
            INSERT INTO stabilized_knowledge
            (id, text, attractor_cos, attractor_sin, coherence, timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle_id,
            text[:5000],
            attractor[0],
            attractor[1],
            coherence,
            timestamp,
            source
        ))
        
        self.conn.commit()
        
        # Mise à jour des caches
        self.states.append(stabilized_state)
        self.cycles[cycle_id] = cycle
        
        # Mise à jour de la cohérence globale
        self.update_global_coherence()
        
        return cycle
    
    # -------------------------------------------------
    # RECHERCHE TRIADIQUE
    # -------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[TriadicCycle, float]]:
        """
        Recherche triadique:
        - Convergence du query vers attracteur
        - Mesure de cohérence avec les connaissances stabilisées
        - Tri par distance sur l'attracteur
        """
        
        # Encodage de la requête
        query_state = self.encode_to_triadic(query)
        query_attractor = self.flow.attractor_projection(query_state)
        
        # Calcul des distances triadiques
        results = []
        for cycle_id, cycle in self.cycles.items():
            # Distance sur l'attracteur circulaire
            cos_q, sin_q = query_attractor
            cos_c, sin_c = cycle.attractor
            
            # Angle entre les attracteurs
            angle_diff = abs(np.arctan2(sin_q - sin_c, cos_q - cos_c))
            
            # Cohérence de la connaissance
            coherence = self.compute_coherence(cycle.state)
            
            # Score triadique
            score = (1 - angle_diff / np.pi) * coherence
            
            # Facteur de dissipation (les connaissances trop dissipatives sont pénalisées)
            score *= (1 - cycle.state.phi_D)
            
            results.append((cycle, score))
        
        # Tri par score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    # -------------------------------------------------
    # RAISONNEMENT TRIADIQUE
    # -------------------------------------------------
    def reason(self, question: str) -> Dict[str, Any]:
        """
        Raisonnement triadique:
        - Recherche de cycles cohérents
        - Convergence du système global
        - Génération de réponse émergente
        """
        
        # Recherche triadique
        search_results = self.search(question)
        
        if not search_results:
            return {
                "response": "Aucune connaissance cohérente trouvée.",
                "coherence": 0.0,
                "convergence_time": 0,
                "sources": []
            }
        
        # Récupération des connaissances
        cursor = self.conn.cursor()
        sources = []
        knowledge_texts = []
        
        for cycle, score in search_results[:3]:
            cursor.execute(
                "SELECT text FROM stabilized_knowledge WHERE id = ?",
                (cycle.id,)
            )
            row = cursor.fetchone()
            if row:
                knowledge_texts.append(row[0])
                sources.append({
                    "id": cycle.id,
                    "coherence": score,
                    "attractor": cycle.attractor
                })
        
        # Création d'un état triadique global
        combined_state = self.combine_states([c.state for c, _ in search_results[:3]])
        
        # Convergence du système global
        converged_state = self.flow.converge(combined_state)
        
        # Calcul de la cohérence finale
        final_coherence = self.compute_coherence(converged_state)
        
        # Génération de la réponse
        response = self.generate_response(question, knowledge_texts, final_coherence)
        
        return {
            "response": response,
            "coherence": final_coherence,
            "convergence_time": len(converged_state.convergence_history),
            "sources": sources,
            "attractor": self.flow.attractor_projection(converged_state)
        }
    
    # -------------------------------------------------
    # MÉTHODES AUXILIAIRES
    # -------------------------------------------------
    def compute_coherence(self, state: TriadicState) -> float:
        """Calcule la cohérence triadique d'un état"""
        
        # Cohérence basée sur la proximité à l'attracteur
        attractor = self.flow.attractor_projection(state)
        distance = np.sqrt(attractor[0]**2 + attractor[1]**2)
        
        # Cohérence = 1 - distance (normalisée)
        coherence = max(0, min(1, 1 - distance))
        
        # Facteur de dissipation
        coherence *= (1 - state.phi_D)
        
        return coherence
    
    def combine_states(self, states: List[TriadicState]) -> TriadicState:
        """Combine plusieurs états triadiques"""
        
        if not states:
            return TriadicState(phi_M=np.zeros(384), phi_C=np.zeros(384), phi_D=0.0)
        
        # Moyenne pondérée par stabilité
        total_weight = sum(s.stability + 1e-6 for s in states)
        
        combined_M = sum(s.phi_M * (s.stability + 1e-6) for s in states) / total_weight
        combined_C = sum(s.phi_C * (s.stability + 1e-6) for s in states) / total_weight
        combined_D = sum(s.phi_D * (s.stability + 1e-6) for s in states) / total_weight
        
        return TriadicState(
            phi_M=combined_M,
            phi_C=combined_C,
            phi_D=combined_D
        )
    
    def generate_response(self, question: str, knowledge: List[str], coherence: float) -> str:
        """Génère une réponse à partir des connaissances triadiques"""
        
        if not knowledge:
            return "Aucune connaissance pertinente trouvée."
        
        # Sélection des passages les plus cohérents
        response_parts = []
        for text in knowledge[:3]:
            response_parts.append(text[:500])
        
        # Indicateur de cohérence
        coherence_indicator = "✓" if coherence > 0.7 else "⚠" if coherence > 0.3 else "✗"
        coherence_msg = f"\n\n[Coherence: {coherence_indicator} {coherence:.2f}]"
        
        response = "\n\n---\n\n".join(response_parts) + coherence_msg
        
        return response
    
    def update_global_coherence(self):
        """Met à jour la cohérence globale du système"""
        
        if not self.states:
            self.global_coherence = 0.0
            return
        
        coherence_sum = sum(self.compute_coherence(s) for s in self.states)
        self.global_coherence = coherence_sum / len(self.states)
        
        # Taux de dissipation moyen
        self.dissipation_rate = sum(s.phi_D for s in self.states) / len(self.states)
    
    # -------------------------------------------------
    # EXTRACTION DE FICHIERS
    # -------------------------------------------------
    def extract_text_from_file(self, file) -> str:
        """Extrait le texte d'un fichier uploadé"""
        
        text = ""
        filename = file.name.lower()
        
        try:
            if filename.endswith('.txt') or file.type == 'text/plain':
                text = file.read().decode('utf-8')
            elif filename.endswith('.pdf') or 'pdf' in file.type:
                pdf = PyPDF2.PdfReader(file)
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
            elif filename.endswith('.docx') or 'word' in file.type:
                doc = docx.Document(file)
                for p in doc.paragraphs:
                    text += p.text + "\n"
            else:
                text = file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Erreur d'extraction: {e}")
            text = ""
        
        return text
    
    def learn_document(self, uploaded_file) -> List[TriadicCycle]:
        """Apprend un document entier"""
        
        text = self.extract_text_from_file(uploaded_file)
        if not text.strip():
            return []
        
        # Segmentation en cycles triadiques
        sections = self.segment_into_cycles(text)
        
        cycles = []
        for section in sections:
            cycle = self.learn(section, source=uploaded_file.name)
            cycles.append(cycle)
        
        return cycles
    
    def segment_into_cycles(self, text: str) -> List[str]:
        """Segmente un texte en cycles triadiques"""
        
        # Découpage naturel
        sections = re.split(r'\n\s*\n|\n---\n', text)
        
        # Filtrage des sections significatives
        cycles = [s.strip() for s in sections if len(s.strip()) > 100]
        
        if not cycles:
            cycles = [text[:1000]]
        
        return cycles
    
    # -------------------------------------------------
    # VISUALISATIONS TRIADIQUES
    # -------------------------------------------------
    def get_attractor_map(self) -> pd.DataFrame:
        """Génère une carte des attracteurs"""
        
        data = []
        for cycle_id, cycle in self.cycles.items():
            data.append({
                "id": cycle_id[:8],
                "cos": cycle.attractor[0],
                "sin": cycle.attractor[1],
                "coherence": self.compute_coherence(cycle.state),
                "phase": cycle.phase
            })
        
        return pd.DataFrame(data)
    
    def get_convergence_path(self, cycle_id: str) -> List[float]:
        """Récupère le chemin de convergence d'un cycle"""
        
        cycle = self.cycles.get(cycle_id)
        if cycle:
            return cycle.state.convergence_history
        return []
    
    def get_global_triad(self) -> Dict[str, float]:
        """Retourne l'état triadique global"""
        
        return {
            "phi_M": float(np.mean([s.phi_M[0] if len(s.phi_M) > 0 else 0 for s in self.states])),
            "phi_C": float(np.mean([s.phi_C[0] if len(s.phi_C) > 0 else 0 for s in self.states])),
            "phi_D": self.dissipation_rate,
            "coherence": self.global_coherence
        }
    
    def stats(self) -> Dict[str, Any]:
        """Statistiques triadiques"""
        
        return {
            "cycles": len(self.cycles),
            "states": len(self.states),
            "global_coherence": self.global_coherence,
            "dissipation_rate": self.dissipation_rate,
            "stable_cycles": sum(1 for c in self.cycles.values() if c.is_stable())
        }

# ==========================================
# APPLICATION STREAMLIT TRIADIQUE
# ==========================================
def main():
    st.set_page_config(
        page_title="Oracle TTU-MC³",
        page_icon="🌀",
        layout="wide"
    )
    
    # Initialisation de l'oracle
    @st.cache_resource
    def get_oracle():
        return TTUOracle()
    
    oracle = get_oracle()
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .triadic-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .coherence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .coherence-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
    }
    .coherence-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.markdown("""
    <div class="triadic-title">
        <h1>🌀 Oracle TTU-MC³</h1>
        <p>Mémoire Triadique | Flot Dynamique | Attracteur Cyclique</p>
        <p style="font-size: 0.8em;">Φ = (Φ_M, Φ_C, Φ_D) → Cercle Unité</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar: État triadique global
    with st.sidebar:
        st.header("🌀 État Triadique Global")
        
        stats = oracle.stats()
        triad = oracle.get_global_triad()
        
        # Métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cycles", stats["cycles"])
            st.metric("Cohérence", f"{triad['coherence']:.3f}")
        with col2:
            st.metric("Stables", stats["stable_cycles"])
            st.metric("Dissipation", f"{triad['phi_D']:.3f}")
        
        # Visualisation de l'état triadique
        fig = go.Figure(data=[
            go.Bar(
                x=['Φ_M', 'Φ_C', 'Φ_D'],
                y=[triad['phi_M'], triad['phi_C'], triad['phi_D']],
                marker_color=['#4CAF50', '#2196F3', '#FF5722']
            )
        ])
        fig.update_layout(
            title="État Triadique",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Statistiques
        st.subheader("📊 Statistiques")
        st.json(stats)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌀 Apprentissage Triadique",
        "🔍 Recherche Triadique",
        "🎯 Carte des Attracteurs",
        "📈 Dynamique Triadique"
    ])
    
    with tab1:
        st.header("Apprentissage Triadique")
        st.markdown("""
        L'apprentissage suit le flot triadique:
        1. **Φ_M (Mémoire)** : Encodage sémantique
        2. **Φ_C (Cohérence)** : Convergence vers attracteur
        3. **Φ_D (Dissipation)** : Compression adaptative
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Texte")
            texte = st.text_area("Entrez un texte à apprendre", height=200)
            if st.button("🌀 Apprendre (Flot Triadique)", type="primary"):
                if texte.strip():
                    with st.spinner("Convergence vers l'attracteur..."):
                        cycle = oracle.learn(texte)
                        st.success(f"✅ Cycle triadique appris: {cycle.id[:8]}")
                        st.info(f"Cohérence: {oracle.compute_coherence(cycle.state):.3f}")
                        
                        # Visualisation de la convergence
                        if cycle.state.convergence_history:
                            fig = go.Figure(data=[
                                go.Scatter(
                                    y=cycle.state.convergence_history,
                                    mode='lines+markers',
                                    name='Stabilité'
                                )
                            ])
                            fig.update_layout(
                                title="Convergence vers l'attracteur",
                                xaxis_title="Itération",
                                yaxis_title="Stabilité"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Veuillez entrer un texte.")
        
        with col2:
            st.subheader("📁 Document")
            uploaded_file = st.file_uploader(
                "Choisissez un fichier",
                type=['txt', 'pdf', 'docx', 'csv']
            )
            if uploaded_file and st.button("📚 Apprendre le document"):
                with st.spinner("Segmentation en cycles triadiques..."):
                    cycles = oracle.learn_document(uploaded_file)
                    st.success(f"✅ {len(cycles)} cycles triadiques appris")
                    
                    # Distribution des cohérences
                    coherences = [oracle.compute_coherence(c.state) for c in cycles]
                    fig = go.Figure(data=[
                        go.Histogram(x=coherences, nbinsx=20)
                    ])
                    fig.update_layout(
                        title="Distribution des cohérences",
                        xaxis_title="Cohérence",
                        yaxis_title="Fréquence"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Recherche Triadique")
        st.markdown("""
        La recherche simule la convergence de la requête vers l'attracteur,
        puis mesure la cohérence avec les connaissances stabilisées.
        """)
        
        question = st.text_input("💭 Posez votre question")
        
        col1, col2 = st.columns([2, 1])
        
        if st.button("🌀 Interroger l'Oracle", type="primary"):
            if question.strip():
                with st.spinner("Convergence triadique en cours..."):
                    result = oracle.reason(question)
                    
                    # Affichage de la cohérence
                    coherence = result["coherence"]
                    if coherence > 0.7:
                        st.markdown(f'<div class="coherence-high">📊 Cohérence: {coherence:.3f} (Élevée)</div>', unsafe_allow_html=True)
                    elif coherence > 0.3:
                        st.markdown(f'<div class="coherence-medium">📊 Cohérence: {coherence:.3f} (Moyenne)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="coherence-low">📊 Cohérence: {coherence:.3f} (Faible)</div>', unsafe_allow_html=True)
                    
                    # Réponse
                    st.markdown("### 🌀 Réponse Émergente")
                    st.write(result["response"])
                    
                    # Métadonnées
                    with st.expander("📊 Métadonnées triadiques"):
                        st.write(f"**Temps de convergence:** {result['convergence_time']} itérations")
                        st.write(f"**Attracteur final:** cos={result['attractor'][0]:.3f}, sin={result['attractor'][1]:.3f}")
                        
                        if result["sources"]:
                            st.write("**Sources:**")
                            for src in result["sources"]:
                                st.write(f"- {src['id']} (cohérence: {src['coherence']:.3f})")
            else:
                st.warning("Veuillez entrer une question.")
    
    with tab3:
        st.header("Carte des Attracteurs")
        st.markdown("""
        Visualisation des connaissances stabilisées sur l'attracteur circulaire.
        Chaque point représente un cycle triadique dont la position est donnée par
        (cos θ, sin θ) sur le cercle unité.
        """)
        
        df = oracle.get_attractor_map()
        
        if not df.empty:
            fig = go.Figure()
            
            # Cercle unité
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Attracteur (cercle unité)'
            ))
            
            # Points de connaissance
            fig.add_trace(go.Scatter(
                x=df['cos'],
                y=df['sin'],
                mode='markers+text',
                marker=dict(
                    size=df['coherence'] * 30,
                    color=df['coherence'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cohérence")
                ),
                text=df['id'],
                textposition="top center",
                name='Cycles triadiques'
            ))
            
            fig.update_layout(
                title="Carte des Attracteurs",
                xaxis_title="cos θ",
                yaxis_title="sin θ",
                xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-1.2, 1.2]),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des attracteurs
            st.subheader("Cycles triadiques")
            st.dataframe(df)
        else:
            st.info("Aucun cycle triadique appris. Commencez par apprendre des textes.")
    
    with tab4:
        st.header("Dynamique Triadique")
        
        # État triadique global
        triad = oracle.get_global_triad()
        
        # Visualisation 3D de l'espace triadique
        fig = go.Figure(data=[
            go.Scatter3d(
                x=[triad['phi_M']],
                y=[triad['phi_C']],
                z=[triad['phi_D']],
                mode='markers',
                marker=dict(size=20, color='red'),
                name='État global'
            )
        ])
        
        # Ajout des cycles
        for cycle_id, cycle in list(oracle.cycles.items())[:50]:
            state = cycle.state
            fig.add_trace(go.Scatter3d(
                x=[state.phi_M[0] if len(state.phi_M) > 0 else 0],
                y=[state.phi_C[0] if len(state.phi_C) > 0 else 0],
                z=[state.phi_D],
                mode='markers',
                marker=dict(size=5, color='blue'),
                name=cycle_id[:8],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Espace Triadique Φ = (Φ_M, Φ_C, Φ_D)",
            scene=dict(
                xaxis_title="Φ_M (Mémoire)",
                yaxis_title="Φ_C (Cohérence)",
                zaxis_title="Φ_D (Dissipation)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Historique de convergence (si disponible)
        if oracle.states:
            st.subheader("Historique des Convergences")
            
            # Derniers chemins de convergence
            convergence_data = []
            for cycle_id, cycle in list(oracle.cycles.items())[:10]:
                history = cycle.state.convergence_history
                if history:
                    convergence_data.append({
                        "cycle": cycle_id[:8],
                        "iterations": list(range(len(history))),
                        "stability": history
                    })
            
            if convergence_data:
                fig = go.Figure()
                for data in convergence_data:
                    fig.add_trace(go.Scatter(
                        x=data["iterations"],
                        y=data["stability"],
                        mode='lines',
                        name=data["cycle"]
                    ))
                
                fig.update_layout(
                    title="Chemins de Convergence",
                    xaxis_title="Itération",
                    yaxis_title="Stabilité",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()