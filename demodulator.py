import re

class Demodulator:
    def __init__(self):
        # On définit des zones de saut de phase pour le français
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        
    def decode_stream(self, m_history):
        tokens = []
        for i in range(1, len(m_history)):
            # On mesure la variation relative (accélération de phase)
            delta = abs(m_history[i] - m_history[i-1])
            
            # Filtre de protection : si le delta est trop grand, on le normalise
            if delta > 26: delta = delta % 26
            
            # Transcription en lettres (on évite les chiffres)
            if 0.1 <= delta <= 26:
                char = self.alphabet[int(delta) % 26]
                tokens.append(char)
            elif delta > 26:
                tokens.append(" ") # Grand saut = Espace

        # NETTOYAGE SYNTAXIQUE (Inspiré de la TST)
        text = "".join(tokens)
        # Supprime les répétitions de plus de 2 lettres
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        # Force l'insertion d'espaces tous les 4-6 caractères pour créer des "mots"
        text = re.sub(r'(\w{5})', r'\1 ', text)
        
        return text.capitalize().strip()[:120] + "."
