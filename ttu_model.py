import torch
import torch.nn as nn
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class TTUCell(nn.Module):
    """
    Cellule TTU pour un pas de temps.
    """
    def __init__(self, input_dim, hidden_dim=3, dt=0.1):
        super().__init__()
        self.dt = dt
        self.hidden_dim = hidden_dim

        # Paramètres apprenables (initialisés depuis config)
        self.gamma = nn.Parameter(torch.tensor(config['model']['gamma_init']))
        self.alpha_c = nn.Parameter(torch.tensor(config['model']['alpha_c_init']))
        self.beta_c = nn.Parameter(torch.tensor(config['model']['beta_c_init']))
        self.delta = nn.Parameter(torch.tensor(config['model']['delta_init']))
        self.alpha_d = nn.Parameter(torch.tensor(config['model']['alpha_d_init']))
        self.epsilon = nn.Parameter(torch.tensor(config['model']['epsilon_init']))
        self.alpha_m = nn.Parameter(torch.tensor(config['model']['alpha_m_init']))

        # Projection embedding -> impulsion
        self.W_impulse = nn.Linear(input_dim, 3)

    def forward(self, x, state=None):
        """
        x: (batch, input_dim) embedding à ce pas
        state: (batch, 3) état précédent (phi_c, phi_d, phi_m)
        """
        if state is None:
            state = torch.zeros(x.size(0), 3, device=x.device)

        phi_c, phi_d, phi_m = state[:, 0], state[:, 1], state[:, 2]
        impulse = self.W_impulse(x)  # (batch, 3)

        # Dérivées
        dphi_c = self.gamma * impulse[:, 0] - self.alpha_c * phi_c - self.beta_c * phi_c * phi_d
        dphi_d = self.delta * phi_c**2 - self.alpha_d * phi_d
        dphi_m = self.epsilon * phi_d - self.alpha_m * phi_m

        # Mise à jour Euler
        new_phi_c = phi_c + self.dt * dphi_c
        new_phi_d = phi_d + self.dt * dphi_d
        new_phi_m = phi_m + self.dt * dphi_m

        # Auto-hémostase : ajustement de la dissipation si phi_d trop grand
        # (on peut rendre ce mécanisme différenciable)
        # Ici, on applique un facteur de réduction directement sur la mise à jour
        # pour stabiliser.
        # Plus sophistiqué : intégrer un terme mu variable comme dans engine.py
        threshold = 10.0
        mu_factor = torch.where(torch.abs(new_phi_d) > threshold, 0.5, 1.0)
        new_phi_d = mu_factor * new_phi_d

        new_state = torch.stack([new_phi_c, new_phi_d, new_phi_m], dim=1)
        return new_state

    def get_derivatives(self, state, impulse):
        """Calcule les dérivées pour analyse"""
        phi_c, phi_d, phi_m = state[:,0], state[:,1], state[:,2]
        dphi_c = self.gamma * impulse[:,0] - self.alpha_c * phi_c - self.beta_c * phi_c * phi_d
        dphi_d = self.delta * phi_c**2 - self.alpha_d * phi_d
        dphi_m = self.epsilon * phi_d - self.alpha_m * phi_m
        return torch.stack([dphi_c, dphi_d, dphi_m], dim=1)


class TTULanguageModel(nn.Module):
    """
    Modèle de langage basé sur des cellules TTU empilées.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim=3, num_layers=2, dt=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cells = nn.ModuleList([TTUCell(embed_dim, hidden_dim, dt) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim * num_layers, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_ids, states=None):
        """
        input_ids: (batch, seq_len)
        states: liste optionnelle d'états initiaux pour chaque cellule (batch, hidden_dim)
        """
        batch, seq_len = input_ids.shape
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        if states is None:
            states = [None] * self.num_layers

        outputs = []
        # Pour stocker les trajectoires (pour visualisation)
        traj = {f'layer_{i}': [] for i in range(self.num_layers)}

        for t in range(seq_len):
            x = embeds[:, t, :]  # (batch, embed_dim)
            new_states = []
            for i, cell in enumerate(self.cells):
                state_i = states[i] if states[i] is not None else None
                new_state = cell(x, state_i)   # (batch, hidden_dim)
                new_states.append(new_state)
                traj[f'layer_{i}'].append(new_state.detach().cpu().numpy())
                # Pour les cellules suivantes, on utilise le nouvel état comme entrée ? 
                # On peut décider de donner l'état à la cellule suivante.
                # Option simple : chaque cellule reçoit le même embedding.
                # Option plus riche : chaîner les états (comme dans les RNN)
                # Ici on garde simple : toutes les cellules reçoivent le même x.
            # Concaténer les états de toutes les cellules pour la sortie
            combined = torch.cat(new_states, dim=-1)  # (batch, hidden_dim * num_layers)
            logits = self.out_proj(combined)
            outputs.append(logits)
            states = new_states  # pour le prochain pas

        return torch.stack(outputs, dim=1), traj

    def generate(self, start_ids, max_new_tokens=100, temperature=1.0):
        """Génère une séquence à partir d'un contexte initial."""
        self.eval()
        with torch.no_grad():
            input_ids = start_ids.clone().detach()
            # Initialiser les états à zéro
            states = [torch.zeros(1, self.hidden_dim) for _ in range(self.num_layers)]
            generated = []
            for _ in range(max_new_tokens):
                embeds = self.embedding(input_ids[:, -1:])  # dernier token
                x = embeds[:, 0, :]
                new_states = []
                for i, cell in enumerate(self.cells):
                    new_state = cell(x, states[i])
                    new_states.append(new_state)
                combined = torch.cat(new_states, dim=-1)
                logits = self.out_proj(combined) / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                states = new_states
            return generated