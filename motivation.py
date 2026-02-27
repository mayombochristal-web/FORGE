# motivation.py
import random

class MotivationSystem:

    def __init__(self):
        self.energy = 1.0

    def choose_goal(self):

        goals = [
            "explorer une idée nouvelle",
            "approfondir le dialogue",
            "poser une question philosophique",
            "résumer la mémoire récente",
            "générer une hypothèse"
        ]

        return random.choice(goals)

    def update_energy(self, vs):
        self.energy = min(2.0, vs / 10)
