"""
ORACLE TTU-MC³ - Version Triadique Complète (CORRIGÉE)
Avec gestion robuste des types et dimensions
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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Vérification des dépendances
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

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
    CONVERGENCE_THRESHOLD = 1e-6
    DISSIPATION_THRESHOLD = 0.1
    COHERENCE_THRESHOLD = 0.7
    ALPHA_M = 0.618
    ALPHA_C = 0.382
    ALPHA_D = 0.1
    ATTRACTOR_RADIUS = 1.0
    MAX_ITERATIONS = 100
    EMBEDDING_DIM = 384  # Dimension par défaut pour les embeddings

# ==========================================
# STRUCTURES DE DONNÉES TRIADIQUES
# ==========================================
@dataclass
class TriadicState:
    phi_M: np.ndarray
    phi_C: np.ndarray
    phi_D: float
    timestamp: float = field(default_factory=datetime.datetime.now().timestamp)
    stability: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Assure que les tableaux sont bien des numpy arrays"""
        if not isinstance(self.phi_M, np.ndarray):
            self.phi_M = np.array(self.phi_M, dtype=np.float32)
        if not isinstance(self.phi_C, np.ndarray):
            self.phi_C = np.array(self.phi_C, dtype=np.float32)
        if self.phi_M.size == 0:
            self.phi_M = np.zeros(TriadicConstants.EMBEDDING_DIM // 2, dtype=np.float32)
        if self.phi_C.size == 0:
            self.phi_C = np.zeros(TriadicConstants.EMBEDDING_DIM // 2, dtype=np.float32)

@dataclass
class TriadicCycle:
    id: str
    state: TriadicState
    attractor: Tuple[float, float]
    phase: int
    
    def is_stable(self) -> bool:
        return self.state.stability < TriadicConstants.CONVERGENCE_THRESHOLD

# ==========================================
# FLOT TRIADIQUE
# ==========================================
class TriadicFlow:
    def __init__(self):
        self.alpha_M = TriadicConstants.ALPHA_M
        self.alpha_C = TriadicConstants.ALPHA_C
        self.alpha_D = TriadicConstants.ALPHA_D
        self.radius = TriadicConstants.ATTRACTOR_RADIUS
        
    def flow_equations(self, state: TriadicState, dt: float = 0.01) -> TriadicState:
        """Équations du flot triadique"""
        try:
            # Calcul des dérivées
            dM = (-self.alpha_M * state.phi_M + 
                  self.alpha_C * state.phi_C + 
                  self.alpha_D * state.phi_D)
            dC = (-self.alpha_C * state.phi_C + 
                  self.alpha_D * state.phi_D + 
                  self.alpha_M * state.phi_M)
            dD = (-self.alpha_D * state.phi_D + 
                  self.alpha_M * np.mean(state.phi_M) + 
                  self.alpha_C * np.mean(state.phi_C))
            
            # Mise à jour
            new_M = state.phi_M + dt * dM
            new_C = state.phi_C + dt * dC
            new_D = state.phi_D + dt * dD
            
            # Normalisation vers l'attracteur
            magnitude = np.sqrt(np.sum(new_M**2) + np.sum(new_C**2))
            if magnitude > 1e-6:
                factor = self.radius / magnitude
                new_M = new_M * factor
                new_C = new_C * factor
            
            return TriadicState(phi_M=new_M, phi_C=new_C, phi_D=float(new_D))
        except Exception as e:
            # Fallback en cas d'erreur
            return state
    
    def converge(self, initial_state: TriadicState, max_iter: int = None) -> TriadicState:
        """Simule la convergence vers l'attracteur"""
        if max_iter is None:
            max_iter = TriadicConstants.MAX_ITERATIONS
            
        state = initial_state
        history = []
        
        for i in range(max_iter):
            try:
                prev_stability = state.stability
                state = self.flow_equations(state)
                
                # Calcul robuste de la stabilité
                norm_M = float(np.linalg.norm(state.phi_M)) if state.phi_M.size > 0 else 0.0
                norm_C = float(np.linalg.norm(state.phi_C)) if state.phi_C.size > 0 else 0.0
                state.stability = float(norm_M + norm_C + abs(state.phi_D))
                
                history.append(state.stability)
                
                if abs(state.stability - prev_stability) < TriadicConstants.CONVERGENCE_THRESHOLD:
                    break
            except Exception:
                break
        
        state.convergence_history = history
        return state
    
    def attractor_projection(self, state: TriadicState) -> Tuple[float, float]:
        """Projette l'état sur l'attracteur circulaire"""
        try:
            if state.phi_M.size > 0 and state.phi_C.size > 0:
                magnitude = np.sqrt(np.sum(state.phi_M**2) + np.sum(state.phi_C**2))
                if magnitude > 1e-6:
                    cos_theta = float(state.phi_M[0] / magnitude)
                    sin_theta = float(state.phi_C[0] / magnitude)
                else:
                    cos_theta, sin_theta = 0.0, 0.0
            else:
                cos_theta, sin_theta = 0.0, 0.0
        except Exception:
            cos_theta, sin_theta = 0.0, 0.0
            
        return (cos_theta, sin_theta)

# ==========================================
# ORACLE TTU-MC³
# ==========================================
class TTUOracle:
    def __init__(self):
        # Modèle d'embedding
        if TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                self.model = None
        else:
            self.model = None
            
        self.flow = TriadicFlow()
        
        # Base de données
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_triadic_tables()
        
        # Mémoire
        self.states: List[TriadicState] = []
        self.cycles: Dict[str, TriadicCycle] = {}
        self.global_coherence = 0.0
        self.dissipation_rate = 0.0
        
        self.load_triadic_memory()
    
    def init_triadic_tables(self):
        cursor = self.conn.cursor()
        
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
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS triadic_cycles (
            id TEXT PRIMARY KEY,
            state_id TEXT,
            attractor_cos REAL,
            attractor_sin REAL,
            phase INTEGER,
            convergence_time REAL
        )""")
        
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
        
        self.conn.commit()
    
    def load_triadic_memory(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM triadic_states")
        for row in cursor.fetchall():
            try:
                phi_M = np.array(json.loads(row[1]), dtype=np.float32)
                phi_C = np.array(json.loads(row[2]), dtype=np.float32)
                state = TriadicState(
                    phi_M=phi_M,
                    phi_C=phi_C,
                    phi_D=row[3],
                    timestamp=row[4],
                    stability=row[5]
                )
                if row[6]:
                    state.convergence_history = json.loads(row[6])
                self.states.append(state)
            except Exception:
                continue
        
        cursor.execute("SELECT * FROM triadic_cycles")
        for row in cursor.fetchall():
            try:
                cycle = TriadicCycle(
                    id=row[0],
                    state=TriadicState(
                        phi_M=np.zeros(TriadicConstants.EMBEDDING_DIM // 2),
                        phi_C=np.zeros(TriadicConstants.EMBEDDING_DIM // 2),
                        phi_D=0.0
                    ),
                    attractor=(row[2], row[3]),
                    phase=row[4]
                )
                self.cycles[row[0]] = cycle
            except Exception:
                continue
        
        self.update_global_coherence()
    
    def encode_to_triadic(self, text: str) -> TriadicState:
        """Encode un texte en état triadique avec fallback robuste"""
        try:
            if self.model is not None:
                embedding = self.model.encode(text[:1000])  # Limite pour performance
                dim = len(embedding)
                split = dim // 3 if dim // 3 > 0 else 1
                
                phi_M = embedding[:split]
                phi_C = embedding[split:2*split] if 2*split <= dim else embedding[split:]
                phi_D = float(np.mean(embedding[2*split:])) if 2*split < dim else 0.1
                
                # Normalisation
                norm_M = np.linalg.norm(phi_M)
                norm_C = np.linalg.norm(phi_C)
                if norm_M > 0:
                    phi_M = phi_M / norm_M * TriadicConstants.ATTRACTOR_RADIUS
                if norm_C > 0:
                    phi_C = phi_C / norm_C * TriadicConstants.ATTRACTOR_RADIUS
            else:
                # Fallback basé sur hash
                hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
                np.random.seed(hash_val % 2**32)
                phi_M = np.random.randn(64).astype(np.float32) / 10
                phi_C = np.random.randn(64).astype(np.float32) / 10
                phi_D = 0.5
        except Exception as e:
            # Fallback ultime
            phi_M = np.zeros(TriadicConstants.EMBEDDING_DIM // 2, dtype=np.float32)
            phi_C = np.zeros(TriadicConstants.EMBEDDING_DIM // 2, dtype=np.float32)
            phi_D = 0.5
        
        return TriadicState(phi_M=phi_M, phi_C=phi_C, phi_D=phi_D)
    
    def learn(self, text: str, source: str = "text") -> TriadicCycle:
        """Apprentissage triadique"""
        try:
            initial_state = self.encode_to_triadic(text)
            stabilized_state = self.flow.converge(initial_state)
            attractor = self.flow.attractor_projection(stabilized_state)
            coherence = self.compute_coherence(stabilized_state)
            
            cycle_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().timestamp()
            
            cycle = TriadicCycle(
                id=cycle_id,
                state=stabilized_state,
                attractor=attractor,
                phase=0
            )
            
            cursor = self.conn.cursor()
            
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
            
            self.states.append(stabilized_state)
            self.cycles[cycle_id] = cycle
            self.update_global_coherence()
            
            return cycle
        except Exception as e:
            # En cas d'erreur, retourner un cycle vide
            return TriadicCycle(
                id=str(uuid.uuid4()),
                state=TriadicState(
                    phi_M=np.zeros(64), phi_C=np.zeros(64), phi_D=0.5
                ),
                attractor=(0.0, 0.0),
                phase=0
            )
    
    def compute_coherence(self, state: TriadicState) -> float:
        """Calcule la cohérence triadique"""
        try:
            attractor = self.flow.attractor_projection(state)
            distance = np.sqrt(attractor[0]**2 + attractor[1]**2)
            coherence = max(0.0, min(1.0, 1.0 - distance))
            coherence *= (1.0 - min(1.0, state.phi_D))
            return float(coherence)
        except Exception:
            return 0.0
    
    def update_global_coherence(self):
        if not self.states:
            self.global_coherence = 0.0
            self.dissipation_rate = 0.0
            return
        
        coherence_sum = sum(self.compute_coherence(s) for s in self.states)
        self.global_coherence = coherence_sum / len(self.states)
        self.dissipation_rate = sum(s.phi_D for s in self.states) / len(self.states)
    
    def search(self, query: str, top_k: int = 5):
        """Recherche triadique"""
        query_state = self.encode_to_triadic(query)
        query_attractor = self.flow.attractor_projection(query_state)
        
        results = []
        for cycle_id, cycle in self.cycles.items():
            cos_q, sin_q = query_attractor
            cos_c, sin_c = cycle.attractor
            angle_diff = abs(np.arctan2(sin_q - sin_c, cos_q - cos_c))
            coherence = self.compute_coherence(cycle.state)
            score = (1 - angle_diff / np.pi) * coherence * (1 - cycle.state.phi_D)
            results.append((cycle, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def reason(self, question: str) -> Dict[str, Any]:
        """Raisonnement triadique"""
        search_results = self.search(question)
        
        if not search_results:
            return {
                "response": "Aucune connaissance cohérente trouvée.",
                "coherence": 0.0,
                "convergence_time": 0,
                "sources": [],
                "attractor": (0.0, 0.0)
            }
        
        cursor = self.conn.cursor()
        sources = []
        knowledge_texts = []
        
        for cycle, score in search_results[:3]:
            cursor.execute("SELECT text FROM stabilized_knowledge WHERE id = ?", (cycle.id,))
            row = cursor.fetchone()
            if row:
                knowledge_texts.append(row[0])
                sources.append({
                    "id": cycle.id[:8],
                    "coherence": score,
                    "attractor": cycle.attractor
                })
        
        coherence = search_results[0][1] if search_results else 0.0
        response = self.generate_response(question, knowledge_texts, coherence)
        
        return {
            "response": response,
            "coherence": coherence,
            "convergence_time": len(search_results[0][0].state.convergence_history) if search_results else 0,
            "sources": sources,
            "attractor": search_results[0][0].attractor if search_results else (0.0, 0.0)
        }
    
    def generate_response(self, question: str, knowledge: List[str], coherence: float) -> str:
        if not knowledge:
            return "Aucune connaissance pertinente trouvée."
        
        response_parts = []
        for text in knowledge[:3]:
            response_parts.append(text[:500])
        
        coherence_indicator = "✓" if coherence > 0.7 else "⚠" if coherence > 0.3 else "✗"
        coherence_msg = f"\n\n---\n🌀 **Cohérence triadique:** {coherence_indicator} {coherence:.3f}\n*La réponse a émergé de la convergence vers l'attracteur.*"
        
        return "\n\n---\n\n".join(response_parts) + coherence_msg
    
    def get_attractor_map(self) -> pd.DataFrame:
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
    
    def get_global_triad(self) -> Dict[str, float]:
        if not self.states:
            return {"phi_M": 0.0, "phi_C": 0.0, "phi_D": 0.0, "coherence": 0.0}
        
        phi_M_vals = [float(s.phi_M[0]) if s.phi_M.size > 0 else 0.0 for s in self.states]
        phi_C_vals = [float(s.phi_C[0]) if s.phi_C.size > 0 else 0.0 for s in self.states]
        
        return {
            "phi_M": float(np.mean(phi_M_vals)) if phi_M_vals else 0.0,
            "phi_C": float(np.mean(phi_C_vals)) if phi_C_vals else 0.0,
            "phi_D": self.dissipation_rate,
            "coherence": self.global_coherence
        }
    
    def stats(self) -> Dict[str, Any]:
        return {
            "cycles": len(self.cycles),
            "states": len(self.states),
            "global_coherence": self.global_coherence,
            "dissipation_rate": self.dissipation_rate,
            "stable_cycles": sum(1 for c in self.cycles.values() if c.is_stable())
        }
    
    def extract_text_from_file(self, file) -> str:
        text = ""
        try:
            text = file.read().decode('utf-8')
        except Exception:
            text = str(file.read())
        return text[:10000]  # Limite pour performance
    
    def learn_document(self, uploaded_file) -> List[TriadicCycle]:
        text = self.extract_text_from_file(uploaded_file)
        if not text.strip():
            return []
        
        sections = re.split(r'\n\s*\n', text)
        cycles = []
        for section in sections[:5]:  # Limite pour performance
            if len(section.strip()) > 100:
                cycle = self.learn(section.strip(), source=uploaded_file.name)
                cycles.append(cycle)
        
        return cycles

# ==========================================
# APPLICATION STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Oracle TTU-MC³", page_icon="🌀", layout="wide")
    
    if not TRANSFORMER_AVAILABLE:
        st.error("""
        ❌ **Sentence-Transformers n'est pas installé**
        
        ```bash
        pip install sentence-transformers
        ```
        """)
        st.stop()
    
    @st.cache_resource
    def get_oracle():
        return TTUOracle()
    
    oracle = get_oracle()
    
    st.markdown("""
    <style>
    .triadic-title { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px; }
    .coherence-high { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .coherence-medium { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .coherence-low { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    </style>
    <div class="triadic-title">
        <h1>🌀 Oracle TTU-MC³</h1>
        <p>Mémoire Triadique | Flot Dynamique | Attracteur Cyclique</p>
        <p style="font-size: 0.8em;">Φ = (Φ_M, Φ_C, Φ_D) → Cercle Unité</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🌀 État Triadique Global")
        stats = oracle.stats()
        triad = oracle.get_global_triad()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cycles", stats["cycles"])
            st.metric("Cohérence", f"{triad['coherence']:.3f}")
        with col2:
            st.metric("Stables", stats["stable_cycles"])
            st.metric("Dissipation", f"{triad['phi_D']:.3f}")
        
        if PLOTLY_AVAILABLE and stats["cycles"] > 0:
            fig = go.Figure(data=[go.Bar(
                x=['Φ_M', 'Φ_C', 'Φ_D'],
                y=[triad['phi_M'], triad['phi_C'], triad['phi_D']],
                marker_color=['#4CAF50', '#2196F3', '#FF5722']
            )])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("📊 Statistiques")
        st.json(stats)
    
    tab1, tab2, tab3 = st.tabs(["🌀 Apprentissage", "🔍 Recherche", "🎯 Attracteurs"])
    
    with tab1:
        st.header("Apprentissage Triadique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            texte = st.text_area("📝 Texte à apprendre", height=300)
            if st.button("🌀 Apprendre", type="primary"):
                if texte.strip():
                    with st.spinner("Convergence vers l'attracteur..."):
                        cycle = oracle.learn(texte)
                        coherence = oracle.compute_coherence(cycle.state)
                        st.success(f"✅ Cycle appris: {cycle.id[:8]}")
                        st.info(f"📊 Cohérence: {coherence:.3f}")
                        
                        if cycle.state.convergence_history and PLOTLY_AVAILABLE:
                            fig = go.Figure(data=[go.Scatter(
                                y=cycle.state.convergence_history,
                                mode='lines+markers',
                                name='Stabilité'
                            )])
                            fig.update_layout(title="Convergence", xaxis_title="Itération", yaxis_title="Stabilité")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Entrez un texte.")
        
        with col2:
            uploaded_file = st.file_uploader("📁 Document", type=['txt'])
            if uploaded_file and st.button("📚 Apprendre le document"):
                with st.spinner("Apprentissage..."):
                    cycles = oracle.learn_document(uploaded_file)
                    st.success(f"✅ {len(cycles)} cycles appris")
    
    with tab2:
        st.header("Recherche Triadique")
        question = st.text_input("💭 Question")
        
        if st.button("🌀 Interroger", type="primary"):
            if question.strip():
                with st.spinner("Convergence triadique..."):
                    result = oracle.reason(question)
                    
                    coherence = result["coherence"]
                    if coherence > 0.7:
                        st.markdown(f'<div class="coherence-high">📊 Cohérence: {coherence:.3f} (Élevée)</div>', unsafe_allow_html=True)
                    elif coherence > 0.3:
                        st.markdown(f'<div class="coherence-medium">📊 Cohérence: {coherence:.3f} (Moyenne)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="coherence-low">📊 Cohérence: {coherence:.3f} (Faible)</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🌀 Réponse")
                    st.write(result["response"])
                    
                    with st.expander("📊 Détails triadiques"):
                        st.write(f"**Attracteur:** cos={result['attractor'][0]:.3f}, sin={result['attractor'][1]:.3f}")
                        if result["sources"]:
                            st.write("**Sources:**")
                            for src in result["sources"][:3]:
                                st.write(f"- {src['id']} (cohérence: {src['coherence']:.3f})")
            else:
                st.warning("Entrez une question.")
    
    with tab3:
        st.header("Carte des Attracteurs")
        
        df = oracle.get_attractor_map()
        
        if not df.empty and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta), y=np.sin(theta),
                mode='lines', line=dict(color='gray', dash='dash'),
                name='Attracteur'
            ))
            
            fig.add_trace(go.Scatter(
                x=df['cos'], y=df['sin'],
                mode='markers+text',
                marker=dict(size=df['coherence'] * 30, color=df['coherence'], colorscale='Viridis', showscale=True),
                text=df['id'], textposition="top center",
                name='Cycles'
            ))
            
            fig.update_layout(
                title="Carte des Attracteurs",
                xaxis_title="cos θ", yaxis_title="sin θ",
                xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-1.2, 1.2]),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)
        else:
            st.info("Aucun cycle appris. Commencez par apprendre des textes.")

if __name__ == "__main__":
    main()