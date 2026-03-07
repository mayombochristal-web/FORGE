class ReasoningEngine:
    def build_reasoning(self, question, results):
        response = []
        response.append("ANALYSE DE LA QUESTION")
        response.append(question)
        response.append("\nCONNAISSANCES ASSOCIÉES")
        for score, text in results:
            response.append(f"- {text[:200]}")
        response.append("\nRAISONNEMENT")
        if results:
            response.append(
                "Les connaissances en mémoire indiquent plusieurs corrélations avec la question."
            )
        else:
            response.append("Aucune connaissance pertinente trouvée.")
        response.append("\nCONCLUSION")
        if results:
            response.append(results[0][1])
        return "\n".join(response)