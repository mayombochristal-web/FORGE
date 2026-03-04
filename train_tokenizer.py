tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("tokenizer/tokenizer.model")
vocab_size = tokenizer.get_piece_size()

model = OmegaTTU_LLM(vocab_size=vocab_size)
state_dict = torch.load("model_weights/omega_ttu_llm.pt", map_location=device)
model.load_state_dict(state_dict)