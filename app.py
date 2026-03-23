"""
ORACLE TTU-MC³ - Version Stable avec gestion Plotly robuste
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
from typing import List, Dict, Tuple, Optional, Any

# Vérification Plotly avec fallback
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_FOLDER = "ttu_oracle_memory"
DB_PATH = os.path.join(MEMORY_FOLDER, "ttu_oracle.db")

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

class Config:
    CONVERGENCE_THRESHOLD = 1e-6
    ALPHA_M = 0.618
    ALPHA_C = 0.382
    ALPHA_D = 0.1
    ATTRACTOR_RADIUS = 1.0
    MAX_ITERATIONS = 50
    EMBEDDING_DIM = 64  # Dimension réduite pour performance

# ==========================================
# ÉTAT TRIADIQUE
# ==========================================
class TriadicState:
    def __init__(self, phi_M=None, phi_C=None, phi_D=0.0):
        self.phi_M = list(phi_M) if phi_M else [0.0] * Config.EMBEDDING_DIM
        self.phi_C = list(phi_C) if phi_C else [0.0] * Config.EMBEDDING_DIM
        self.phi_D = float(phi_D)
        self.timestamp = datetime.datetime.now().timestamp()
        self.stability = 0.0
        self.convergence_history = []

# ==========================================
# FLOT TRIADIQUE
# ==========================================
class TriadicFlow:
    def __init__(self):
        self.alpha_M = Config.ALPHA_M
        self.alpha_C = Config.ALPHA_C
        self.alpha_D = Config.ALPHA_D
        self.radius = Config.ATTRACTOR_RADIUS
    
    def _norm(self, vec):
        try:
            return np.sqrt(sum(v * v for v in vec))
        except:
            return 0.0
    
    def _normalize(self, vec):
        norm = self._norm(vec)
        if norm > 1e-6:
            factor = self.radius / norm
            return [v * factor for v in vec]
        return vec
    
    def flow_equations(self, state: TriadicState, dt: float = 0.01) -> TriadicState:
        try:
            dM = [-self.alpha_M * m + self.alpha_C * c + self.alpha_D * state.phi_D 
                  for m, c in zip(state.phi_M, state.phi_C)]
            dC = [-self.alpha_C * c + self.alpha_D * state.phi_D + self.alpha_M * m 
                  for m, c in zip(state.phi_M, state.phi_C)]
            dD = (-self.alpha_D * state.phi_D + 
                  self.alpha_M * sum(state.phi_M)/len(state.phi_M) + 
                  self.alpha_C * sum(state.phi_C)/len(state.phi_C))
            
            new_M = [m + dt * dm for m, dm in zip(state.phi_M, dM)]
            new_C = [c + dt * dc for c, dc in zip(state.phi_C, dC)]
            new_D = state.phi_D + dt * dD
            
            new_M = self._normalize(new_M)
            new_C = self._normalize(new_C)
            
            return TriadicState(phi_M=new_M, phi_C=new_C, phi_D=new_D)
        except:
            return state
    
    def converge(self, initial_state: TriadicState, max_iter: int = None) -> TriadicState:
        if max_iter is None:
            max_iter = Config.MAX_ITERATIONS
        
        state = initial_state
        history = []
        
        for _ in range(max_iter):
            try:
                prev_stability = state.stability
                state = self.flow_equations(state)
                norm_M = self._norm(state.phi_M)
                norm_C = self._norm(state.phi_C)
                state.stability = norm_M + norm_C + abs(state.phi_D)
                history.append(state.stability)
                if abs(state.stability - prev_stability) < Config.CONVERGENCE_THRESHOLD:
                    break
            except:
                break
        
        state.convergence_history = history
        return state
    
    def attractor_projection(self, state: TriadicState) -> Tuple[float, float]:
        try:
            if state.phi_M and state.phi_C:
                mag = self._norm(state.phi_M) + self._norm(state.phi_C)
                if mag > 1e-6:
                    cos_theta = state.phi_M[0] / mag if state.phi_M else 0.0
                    sin_theta = state.phi_C[0] / mag if state.phi_C else 0.0
                else:
                    cos_theta, sin_theta = 0.0, 0.0
            else:
                cos_theta, sin_theta = 0.0, 0.0
        except:
            cos_theta, sin_theta = 0.0, 0.0
        return (cos_theta, sin_theta)

# ==========================================
# ORACLE TTU-MC³
# ==========================================
class TTUOracle:
    def __init__(self):
        self.model = None
        if TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except:
                pass
        
        self.flow = TriadicFlow()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self.states: List[TriadicState] = []
        self.cycles: Dict[str, dict] = {}
        self.global_coherence = 0.0
        self._load_memory()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            text TEXT,
            phi_M TEXT,
            phi_C TEXT,
            phi_D REAL,
            attractor_cos REAL,
            attractor_sin REAL,
            coherence REAL,
            timestamp REAL,
            source TEXT
        )""")
        self.conn.commit()
    
    def _load_memory(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, phi_M, phi_C, phi_D, attractor_cos, attractor_sin FROM knowledge")
        for row in cursor.fetchall():
            try:
                phi_M = json.loads(row[1]) if row[1] else []
                phi_C = json.loads(row[2]) if row[2] else []
                state = TriadicState(phi_M=phi_M, phi_C=phi_C, phi_D=row[3])
                self.cycles[row[0]] = {
                    "id": row[0],
                    "state": state,
                    "attractor": (row[4], row[5])
                }
                self.states.append(state)
            except:
                continue
        self._update_coherence()
    
    def _encode_text(self, text: str) -> Tuple[List[float], List[float], float]:
        try:
            if self.model:
                embedding = self.model.encode(text[:500])
                if len(embedding) > Config.EMBEDDING_DIM:
                    embedding = embedding[:Config.EMBEDDING_DIM]
                elif len(embedding) < Config.EMBEDDING_DIM:
                    embedding = np.pad(embedding, (0, Config.EMBEDDING_DIM - len(embedding)))
            else:
                hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
                np.random.seed(hash_val % 2**32)
                embedding = np.random.randn(Config.EMBEDDING_DIM)
            
            split = Config.EMBEDDING_DIM // 2
            phi_M = embedding[:split].tolist()
            phi_C = embedding[split:].tolist()
            phi_D = float(np.mean(embedding))
            
            norm_M = np.sqrt(sum(v*v for v in phi_M))
            norm_C = np.sqrt(sum(v*v for v in phi_C))
            if norm_M > 0:
                phi_M = [v / norm_M for v in phi_M]
            if norm_C > 0:
                phi_C = [v / norm_C for v in phi_C]
            
            return phi_M, phi_C, phi_D
        except:
            return [0.0] * (Config.EMBEDDING_DIM // 2), [0.0] * (Config.EMBEDDING_DIM // 2), 0.5
    
    def learn(self, text: str, source: str = "text") -> Optional[str]:
        try:
            phi_M, phi_C, phi_D = self._encode_text(text)
            state = TriadicState(phi_M=phi_M, phi_C=phi_C, phi_D=phi_D)
            converged = self.flow.converge(state)
            attractor = self.flow.attractor_projection(converged)
            
            coherence = 1.0 - abs(attractor[0]) - abs(attractor[1])
            coherence = max(0.0, min(1.0, coherence))
            
            cycle_id = str(uuid.uuid4())
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge 
                (id, text, phi_M, phi_C, phi_D, attractor_cos, attractor_sin, coherence, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_id, text[:2000],
                json.dumps(converged.phi_M), json.dumps(converged.phi_C),
                converged.phi_D, attractor[0], attractor[1], coherence,
                datetime.datetime.now().timestamp(), source
            ))
            self.conn.commit()
            
            self.cycles[cycle_id] = {"id": cycle_id, "state": converged, "attractor": attractor}
            self.states.append(converged)
            self._update_coherence()
            return cycle_id
        except:
            return None
    
    def _update_coherence(self):
        if not self.cycles:
            self.global_coherence = 0.0
            return
        coherence_sum = 0.0
        for cycle in self.cycles.values():
            coherence = 1.0 - abs(cycle["attractor"][0]) - abs(cycle["attractor"][1])
            coherence_sum += max(0.0, min(1.0, coherence))
        self.global_coherence = coherence_sum / len(self.cycles)
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        try:
            phi_M_q, phi_C_q, _ = self._encode_text(query)
            q_mag = np.sqrt(sum(v*v for v in phi_M_q) + sum(v*v for v in phi_C_q))
            results = []
            for cycle_id, cycle in self.cycles.items():
                try:
                    dot = sum(a*b for a,b in zip(phi_M_q, cycle["state"].phi_M)) + \
                          sum(a*b for a,b in zip(phi_C_q, cycle["state"].phi_C))
                    mag = self.flow._norm(cycle["state"].phi_M) + self.flow._norm(cycle["state"].phi_C)
                    similarity = dot / (q_mag * mag) if (mag > 0 and q_mag > 0) else 0.0
                    results.append({"id": cycle_id, "similarity": similarity, "attractor": cycle["attractor"]})
                except:
                    continue
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except:
            return []
    
    def reason(self, question: str) -> Dict[str, Any]:
        results = self.search(question)
        if not results:
            return {"response": "Aucune connaissance trouvée.", "coherence": 0.0, "sources": []}
        
        cursor = self.conn.cursor()
        knowledge = []
        for r in results[:2]:
            cursor.execute("SELECT text, coherence FROM knowledge WHERE id = ?", (r["id"],))
            row = cursor.fetchone()
            if row:
                knowledge.append({"text": row[0][:500], "coherence": row[1], "similarity": r["similarity"]})
        
        coherence = knowledge[0]["coherence"] if knowledge else 0.0
        return {
            "response": self._generate_response(question, knowledge, coherence),
            "coherence": coherence,
            "sources": [k["text"][:200] + "..." for k in knowledge]
        }
    
    def _generate_response(self, question: str, knowledge: List[dict], coherence: float) -> str:
        if not knowledge:
            return "Aucune connaissance pertinente."
        parts = [k["text"] for k in knowledge[:2]]
        coherence_ind = "✓" if coherence > 0.6 else "⚠" if coherence > 0.3 else "✗"
        return "\n\n".join(parts) + f"\n\n---\n🌀 **Cohérence:** {coherence_ind} {coherence:.2f}"
    
    def get_stats(self) -> Dict:
        return {
            "cycles": len(self.cycles),
            "coherence": self.global_coherence,
            "stable": sum(1 for s in self.states if s.stability < 0.1)
        }
    
    def get_attractors(self) -> pd.DataFrame:
        data = []
        for cycle_id, cycle in list(self.cycles.items())[:50]:
            data.append({
                "id": cycle_id[:6],
                "cos": cycle["attractor"][0],
                "sin": cycle["attractor"][1],
                "coherence": 1.0 - abs(cycle["attractor"][0]) - abs(cycle["attractor"][1])
            })
        return pd.DataFrame(data)
    
    def get_triad_state(self) -> Dict:
        if not self.states:
            return {"M": 0.0, "C": 0.0, "D": 0.0}
        return {
            "M": np.mean([s.phi_M[0] if s.phi_M else 0 for s in self.states]),
            "C": np.mean([s.phi_C[0] if s.phi_C else 0 for s in self.states]),
            "D": np.mean([s.phi_D for s in self.states])
        }

# ==========================================
# APPLICATION STREAMLIT AVEC GESTION PLOTLY
# ==========================================
def main():
    st.set_page_config(page_title="Oracle TTU-MC³", page_icon="🌀", layout="wide")
    
    @st.cache_resource
    def get_oracle():
        return TTUOracle()
    
    oracle = get_oracle()
    
    # CSS
    st.markdown("""
    <style>
    .title { text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px; }
    .high { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .medium { background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    .low { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    </style>
    <div class="title">
        <h1>🌀 Oracle TTU-MC³</h1>
        <p>Φ = (Φ_M, Φ_C, Φ_D) → Cercle Unité</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec gestion des clés uniques
    with st.sidebar:
        st.header("État Triadique")
        stats = oracle.get_stats()
        triad = oracle.get_triad_state()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cycles", stats["cycles"])
            st.metric("Cohérence", f"{stats['coherence']:.2f}")
        with col2:
            st.metric("Stables", stats["stable"])
            st.metric("Φ_D", f"{triad['D']:.2f}")
        
        # Graphique avec clé unique pour éviter les conflits DOM
        if PLOTLY_AVAILABLE and stats["cycles"] > 0:
            try:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Φ_M', 'Φ_C', 'Φ_D'],
                    y=[triad['M'], triad['C'], triad['D']],
                    marker_color=['#4CAF50', '#2196F3', '#FF5722'],
                    name="État triadique"
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                # Utilisation de key unique pour éviter les conflits
                st.plotly_chart(fig, use_container_width=True, key="sidebar_triad")
            except Exception as e:
                st.warning("Graphique temporairement indisponible")
        
        st.divider()
        st.json(stats)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🌀 Apprentissage", "🔍 Interrogation", "🎯 Carte"])
    
    with tab1:
        st.header("Apprentissage Triadique")
        texte = st.text_area("Texte à apprendre", height=300, 
                            placeholder="Entrez votre texte ici...", key="learn_text")
        
        if st.button("🌀 Apprendre", type="primary", key="learn_btn"):
            if texte.strip():
                with st.spinner("Convergence vers l'attracteur..."):
                    cycle_id = oracle.learn(texte)
                    if cycle_id:
                        st.success(f"✅ Appris: {cycle_id[:8]}")
                        st.info(f"📊 Cohérence globale: {oracle.global_coherence:.3f}")
                        st.rerun()
                    else:
                        st.error("Erreur d'apprentissage")
            else:
                st.warning("Entrez un texte")
    
    with tab2:
        st.header("Interrogation Triadique")
        question = st.text_input("💭 Votre question", key="question_input")
        
        if st.button("🌀 Interroger", type="primary", key="reason_btn"):
            if question.strip():
                with st.spinner("Convergence..."):
                    result = oracle.reason(question)
                    
                    coh = result["coherence"]
                    if coh > 0.6:
                        st.markdown(f'<div class="high">📊 Cohérence: {coh:.2f}</div>', unsafe_allow_html=True)
                    elif coh > 0.3:
                        st.markdown(f'<div class="medium">📊 Cohérence: {coh:.2f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="low">📊 Cohérence: {coh:.2f}</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🌀 Réponse")
                    st.write(result["response"])
                    
                    if result["sources"]:
                        with st.expander("Sources"):
                            for src in result["sources"]:
                                st.write(f"- {src}")
            else:
                st.warning("Entrez une question")
    
    with tab3:
        st.header("Carte des Attracteurs")
        
        df = oracle.get_attractors()
        
        if not df.empty and PLOTLY_AVAILABLE:
            try:
                fig = go.Figure()
                
                # Cercle unité
                theta = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=np.cos(theta), y=np.sin(theta),
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    name='Attracteur',
                    hoverinfo='none'
                ))
                
                # Points de connaissance
                fig.add_trace(go.Scatter(
                    x=df['cos'].tolist(),
                    y=df['sin'].tolist(),
                    mode='markers+text',
                    marker=dict(
                        size=[max(10, min(40, c*30 + 5)) for c in df['coherence']],
                        color=df['coherence'].tolist(),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Cohérence", x=1.02),
                        line=dict(width=1, color='white')
                    ),
                    text=df['id'].tolist(),
                    textposition="top center",
                    textfont=dict(size=10),
                    name='Connaissances',
                    hovertemplate='ID: %{text}<br>Cos: %{x:.2f}<br>Sin: %{y:.2f}<br>Cohérence: %{marker.color:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Projection sur l'Attracteur Circulaire",
                    xaxis_title="cos θ (Φ_M)",
                    yaxis_title="sin θ (Φ_C)",
                    xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1, gridcolor='#eee'),
                    yaxis=dict(range=[-1.2, 1.2], gridcolor='#eee'),
                    height=550,
                    hovermode='closest',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="attractor_map")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Erreur d'affichage: {e}")
        else:
            st.info("Aucune connaissance apprise. Commencez par apprendre des textes.")
    
    # Footer
    st.divider()
    st.caption("🌀 Oracle TTU-MC³ - Théorie Triadique Unifiée | Flot Dynamique vers Attracteur")

if __name__ == "__main__":
    main()