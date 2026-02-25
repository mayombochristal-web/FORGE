class Demodulator:
    def __init__(self):
        # Glossaire Spectral : Mapping des fréquences de courbure
        self.spectral_map = {
            "voyelles": [0.01, 0.45],   # Ondes fluides (a, e, i, o, u)
            "consonnes": [0.45, 1.15],  # Ondes percutantes (s, t, r, n...)
            "complexes": [1.15, 1.95],  # Ondes denses (k, w, x, y, z)
            "espaces": [1.95, 2.80],    # Rupture de phase (Zero-crossing)
            "ponctuation": [2.80, 5.0]  # Choc entropique (. ! ?)
        }

    def decode_stream(self, memory_history):
        substance = []
        for i in range(1, len(memory_history)):
            # Calcul de la "Vitesse de Phase" (la pente entre deux points)
            v_phase = abs(memory_history[i] - memory_history[i-1])
            
            # 1. Zone des Voyelles (Fluidité)
            if self.spectral_map["voyelles"][0] <= v_phase < self.spectral_map["voyelles"][1]:
                val = int(97 + (v_phase * 40) % 5) # Mappe sur a, e, i, o, u
                substance.append(chr([97, 101, 105, 111, 117][val % 5]))
            
            # 2. Zone des Consonnes (Structure)
            elif self.spectral_map["consonnes"][0] <= v_phase < self.spectral_map["consonnes"][1]:
                val = int(98 + (v_phase * 25) % 20)
                substance.append(chr(val))
                
            # 3. Zone des Espaces (Rupture)
            elif self.spectral_map["espaces"][0] <= v_phase < self.spectral_map["espaces"][1]:
                substance.append(" ")
                
            # 4. Zone de Ponctuation (Scellement)
            elif v_phase >= self.spectral_map["ponctuation"][0]:
                substance.append(".")

        # Nettoyage et mise en forme "brillante"
        raw_text = "".join(substance).strip()
        if len(raw_text) < 5:
            return "Dissipation trop lisse... Intensifiez le prompt."
            
        # Capitalisation automatique du premier caractère (Logique de Phase)
        return raw_text.capitalize()[:120]
