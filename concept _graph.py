from collections import Counter

class ConceptGraph:
    def __init__(self):
        self.graph = Counter()

    def update(self, tokens):
        for t in tokens:
            self.graph[t] += 1

    def top(self, n=20):
        return self.graph.most_common(n)