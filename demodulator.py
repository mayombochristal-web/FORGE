class Demodulator:
    def decode_stream(self, memory_history):
        chars = []
        for i in range(1, len(memory_history)):
            diff = abs(memory_history[i] - memory_history[i-1])
            # Isomorphisme : conversion de la courbure en texte
            if 0.5 < diff < 1.2: 
                val = int(97 + (diff * 15) % 26)
                chars.append(chr(val))
            elif diff >= 1.2: 
                chars.append(" ")
        
        result = "".join(chars).strip()
        return result if len(result) > 2 else "Stabilisation en cours..."
