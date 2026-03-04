import streamlit as st
import torch
import sentencepiece as spm
from model import OmegaTTU_LLM

# Configuration de la page
st.set_page_config(page_title="Ω‑TTU LLM Générateur", page_icon="🧠")
st.title("🧠 Ω‑TTU‑LLM‑50M – Générateur de texte")

# Cache pour charger le modèle une seule fois
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Charger le tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/tokenizer.model")
    # Créer le modèle
    model = OmegaTTU_LLM(vocab_size=tokenizer.get_piece_size())
    # Charger les poids (s'ils existent)
    try:
        state_dict = torch.load("model_weights/omega_ttu_llm.pt", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.warning("Poids du modèle non trouvés. Le modèle est initialisé aléatoirement.")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# Fonction de génération
def generate(prompt, max_new_tokens=100, temperature=0.8):
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = input_tensor
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)
        next_token_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        # Stoppe si token de fin (si vous en avez défini un)
        # if next_token.item() == tokenizer.eos_id():
        #     break

    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)

# Interface
prompt = st.text_area("📝 Entrez votre prompt :", height=100)
col1, col2 = st.columns([1, 3])
with col1:
    max_tokens = st.slider("Max tokens", 10, 300, 100)
with col2:
    temperature = st.slider("Température", 0.1, 2.0, 0.8, step=0.1)

if st.button("🚀 Générer"):
    if prompt.strip():
        with st.spinner("Génération en cours..."):
            output = generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        st.markdown("### Réponse :")
        st.write(output)
    else:
        st.warning("Veuillez entrer un prompt.")