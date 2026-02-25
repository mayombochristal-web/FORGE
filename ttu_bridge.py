import numpy as np

class TTU_LLM_Bridge:
    def __init__(self, kernel):
        self.kernel = kernel

    def extract_semantic_vector(self, solution=None):
        """
        Extrait les paramètres sémantiques (température, top_p) à partir
        de la dynamique de cohérence.
        """
        if solution is None:
            solution = self.kernel.solution
        if solution is None:
            return {'temperature': 0.7, 'top_p': 0.9}

        pc = solution.y[1]
        std_pc = np.std(pc)
        # Éviter division par zéro
        pc_range = np.max(pc) - np.min(pc)
        if pc_range < 1e-8:
            temperature = 0.7
        else:
            temperature = 0.1 + (std_pc / pc_range) * 1.4
        temperature = np.clip(temperature, 0.1, 1.5)

        recent_pc = pc[-100:] if len(pc) > 100 else pc
        mean_recent = np.mean(recent_pc)
        top_p = 0.5 + (mean_recent + 2) / 4 * 0.45
        top_p = np.clip(top_p, 0.5, 0.95)

        return {'temperature': float(temperature), 'top_p': float(top_p)}

    def decode_substance_to_prompt(self, substance=None):
        """
        Utilise la substance comme amorce pour le prompt.
        """
        if substance is None:
            substance = self.kernel.substance
        if not substance:
            return "Une fois, dans un univers quantique..."
        return substance[:50]