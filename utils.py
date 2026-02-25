import numpy as np
import plotly.graph_objects as go
import requests
import os
import wikipediaapi

# Initialisation de l'API Wikipédia
wiki = wikipediaapi.Wikipedia(
    language='fr',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='TTU-AI-Chatbot/1.0'
)

def search_wikipedia(query, max_sentences=3):
    """Recherche sur Wikipédia et retourne un résumé."""
    try:
        page = wiki.page(query)
        if page.exists():
            # Prendre les premières phrases du résumé
            summary = page.summary.split('.')
            result = '. '.join(summary[:max_sentences]) + '.'
            return result
        else:
            # Chercher une page par suggestion
            search_url = f"https://fr.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            resp = requests.get(search_url).json()
            if resp['query']['search']:
                title = resp['query']['search'][0]['title']
                page = wiki.page(title)
                if page.exists():
                    summary = page.summary.split('.')
                    result = '. '.join(summary[:max_sentences]) + '.'
                    return result
            return None
    except Exception as e:
        print(f"Erreur Wiki: {e}")
        return None

def plot_trajectory_3d(traj):
    points = np.array(traj).squeeze(1)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='lines+markers',
        marker=dict(size=2, color=np.arange(len(points)), colorscale='Viridis'),
        line=dict(color='darkblue', width=2)
    )])
    fig.update_layout(
        title="Trajectoire TTU - État interne",
        scene=dict(
            xaxis_title='Phi_C (Cohérence)',
            yaxis_title='Phi_D (Dissipation)',
            zaxis_title='Phi_M (Mémoire)'
        )
    )
    return fig

def download_corpus(name="shakespeare"):
    os.makedirs("data", exist_ok=True)
    urls = {
        'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'tiny_shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'wikitext': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt'
    }
    url = urls.get(name, urls['shakespeare'])
    dest = os.path.join('data', f"{name}.txt")
    if not os.path.exists(dest):
        print(f"Téléchargement de {url}...")
        r = requests.get(url)
        with open(dest, 'wb') as f:
            f.write(r.content)
    return dest