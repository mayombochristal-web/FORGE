import re

class Demodulator:
    def __init__(self):
        # Glossaire Spectral Exhaustif
        self.spec = {
            "v": [0.01, 0.50], # Zone des voyelles
            "c": [0.50, 1.30], # Zone des consonnes
            "s": [1.30, 2.50]  # Zone des ruptures (espaces)
        }

    def clean_text(self, text):
        # Suppression des répétitions excessives (max 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        # Simulation de structure de mots par injection d'espaces
        text = re.sub(r'(\w{7})', r'\1 ', text)
        return text

    def decode_stream(self, m_history):
        substance = []
        last_val = 0
        
        for i in range(1, len(m_history)):
            diff = abs(m_history[i] - memory_history[i-1] if 'memory_history' in locals() else m_history[i-1])
            v_phase = abs(diff)
            
            # Anti-stagnation : amplification des micro-variations
            if abs(v_phase - last_val) < 0.001:
                v_phase *= 1.5 
            
            # Transcription Isomorphe
            if self.spec["v"][0] <= v_phase < self.spec["v"][1]:
                char = "aeiouy"[int(v_phase * 100) % 6]
            elif self.spec["c"][0] <= v_phase < self.spec["c"][1]:
                char = "stnrldmpbc"[int(v_phase * 50) % 10]
            elif self.spec["s"][0] <= v_phase < self.spec["s"][1]:
                char = " "
            else: continue
            
            substance.append(char)
            last_val = v_phase

        final_text = self.clean_text("".join(substance))
        if len(final_text) < 3: return "Résonance faible... Intensifiez l'impulsion."
        return final_text.capitalize()[:120] + "."
