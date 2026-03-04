import torch
import sentencepiece as spm
from model import OmegaTTU_LLM

# Charger le tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("tokenizer/tokenizer.model")
vocab_size_tokenizer = tokenizer.get_piece_size()

# Charger le modèle (sans les poids)
model = OmegaTTU_LLM(vocab_size=vocab_size_tokenizer)  # ou charger l'état

# Vérifier la concordance
if model.lm_head.out_features != vocab_size_tokenizer:
    raise ValueError("Incompatibilité : la taille du vocabulaire du modèle ne correspond pas à celle du tokenizer.")
else:
    print("✅ Tokenizer et modèle sont compatibles.")