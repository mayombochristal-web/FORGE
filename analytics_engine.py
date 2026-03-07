class AnalyticsEngine:
    def analyze(self, text):
        words = text.split()
        return {
            "chars": len(text),
            "words": len(words),
            "unique_words": len(set(words))
        }