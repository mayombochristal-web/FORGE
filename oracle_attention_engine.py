class OracleAttentionEngine:

    def __init__(self):

        self.threshold = 12

    def score(self, text):

        length_score = len(text) / 100

        concept_score = text.count("concept")

        score = length_score + concept_score

        return score