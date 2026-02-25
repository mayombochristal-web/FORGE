import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class TTUCell(nn.Module):
    def __init__(self, hidden_dim=3, dt=0.1):
        super().__init__()
        self.dt = dt
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.tensor(config['model']['gamma_init']))
        self.alpha_c = nn.Parameter(torch.tensor(config['model']['alpha_c_init']))
        self.beta_c = nn.Parameter(torch.tensor(config['model']['beta_c_init']))
        self.delta = nn.Parameter(torch.tensor(config['model']['delta_init']))
        self.alpha_d = nn.Parameter(torch.tensor(config['model']['alpha_d_init']))
        self.epsilon = nn.Parameter(torch.tensor(config['model']['epsilon_init']))
        self.alpha_m = nn.Parameter(torch.tensor(config['model']['alpha_m_init']))

    def forward(self, state, impulse):
        phi_c, phi_d, phi_m = state[:,0], state[:,1], state[:,2]
        dphi_c = self.gamma * impulse[:,0] - self.alpha_c * phi_c - self.beta_c * phi_c * phi_d
        dphi_d = self.delta * phi_c**2 - self.alpha_d * phi_d
        dphi_m = self.epsilon * phi_d - self.alpha_m * phi_m
        new_phi_c = phi_c + self.dt * dphi_c
        new_phi_d = phi_d + self.dt * dphi_d
        new_phi_m = phi_m + self.dt * dphi_m
        threshold = 10.0
        mu_factor = torch.where(torch.abs(new_phi_d) > threshold, 0.5, 1.0)
        new_phi_d = mu_factor * new_phi_d
        new_state = torch.stack([new_phi_c, new_phi_d, new_phi_m], dim=1)
        return new_state

class TTULanguageModel(nn.Module):
    def __init__(self, base_model_name="microsoft/DialoGPT-medium", hidden_dim=3, dt=0.1):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.ttu_cell = TTUCell(hidden_dim, dt)
        self.impulse_proj = nn.Linear(self.base_model.config.hidden_size, 3)
        self.logit_modulator = nn.Linear(hidden_dim, self.base_model.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, ttu_state=None):
        outputs = self.base_model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_size, seq_len, _ = hidden_states.shape

        if ttu_state is None:
            ttu_state = torch.zeros(batch_size, self.hidden_dim, device=input_ids.device)

        traj = []
        all_logits = []
        for t in range(seq_len):
            impulse = self.impulse_proj(hidden_states[:, t, :])
            ttu_state = self.ttu_cell(ttu_state, impulse)
            traj.append(ttu_state.detach().cpu().numpy())
            logits = self.base_model.lm_head(hidden_states[:, t, :])
            modulation = self.logit_modulator(ttu_state)
            logits = logits + 0.1 * modulation
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1), ttu_state, traj

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, ttu_state=None, knowledge=None):
        self.eval()
        with torch.no_grad():
            # Ajouter les connaissances si fournies
            if knowledge:
                prompt = prompt + "\n[Contexte: " + knowledge + "]"
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            attention_mask = torch.ones_like(input_ids)
            generated = input_ids
            if ttu_state is None:
                ttu_state = torch.zeros(1, self.hidden_dim)

            traj = []
            for _ in range(max_new_tokens):
                outputs = self.base_model.transformer(input_ids=generated, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state[:, -1, :]
                impulse = self.impulse_proj(last_hidden)
                ttu_state = self.ttu_cell(ttu_state, impulse)
                traj.append(ttu_state.detach().cpu().numpy())
                logits = self.base_model.lm_head(last_hidden)
                modulation = self.logit_modulator(ttu_state)
                logits = logits + 0.1 * modulation
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1,1)], dim=1)

            return self.tokenizer.decode(generated[0], skip_special_tokens=True), ttu_state, traj