import torch
import sentencepiece as spm
from model import OmegaTTU_LLM

# Charger le tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("tokenizer/tokenizer.model")
vocab_size = tokenizer.get_piece_size()

# Créer le modèle avec la bonne dimension
model = OmegaTTU_LLM(vocab_size=vocab_size, dim=384, layers=8, heads=6)
# ... suite de l'entraînement ...