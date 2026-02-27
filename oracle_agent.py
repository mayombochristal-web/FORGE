# oracle_agent.py

from memory import Memory
from operator_L import OperateurL
from motivation import MotivationSystem
from consciousness import Consciousness
import random

class OracleAgent:

    def __init__(self):

        self.memory = Memory()
        self.operator = OperateurL()
        self.motivation = MotivationSystem()
        self.consciousness = Consciousness()

        self.vs = 12.0
        self.identity = "Oracle TTU"

    # pensée autonome
    def autonomous_thought(self):

        goal = self.motivation.choose_goal()
        inner_voice = self.consciousness.internal_monologue(self.memory)

        return f"""
IDENTITÉ : {self.identity}

Pensée interne:
{inner_voice}

Objectif spontané:
{goal}
"""

    def evaluate(self, text):

        ith = len(text)/60
        ancrage = 1.3
        alpha = 0.25

        self.vs = self.operator.vitality(ith, ancrage, alpha)
        self.vs += self.operator.regulate(self.vs)

        self.motivation.update_energy(self.vs)

    def update_memory(self, role, content):
        self.memory.add(role, content)
