class AttentionEngine:

    def score(self,tokens):

        unique=len(set(tokens))

        length=len(tokens)

        score=(unique*0.5)+(length*0.2)

        return score