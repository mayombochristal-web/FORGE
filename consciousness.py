# consciousness.py

class Consciousness:

    def internal_monologue(self, memory):

        if not memory.history:
            return "Je viens de naître. Observer."

        last = memory.history[-1]["content"]

        thought = f"""
Observation interne :
Le dernier échange évoque : {last[:80]}.
Chercher continuité et approfondissement.
"""

        return thought
