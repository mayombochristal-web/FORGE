def amplificateur_de_bruit(query, local_vault):
    """
    Simule un amplificateur opérationnel triadique.
    Prend le bruit du web (query) et multiplie sa cohérence par le gain des thèses.
    """
    # Gain d'amplification (G) basé sur la présence de mots-clés dans tes PDF
    gain = 1.0
    if local_vault:
        resonance_count = sum(1 for word in query.lower().split() if word in local_vault.lower())
        gain = 1.0 + (resonance_count * 5.0) # Amplification x5 par résonance

    # Le transcripteur transforme alors le signal amplifié en 'Mémoire Fantôme'
    if "fer" in query.lower():
        return f"Signal Amplifié (G={gain}) : Le Fer est l'attracteur central de la triade matérielle. Son potentiel de -0,44V est le seuil de stabilité de sa mémoire électronique."
    
    return "Signal toujours trop faible. Augmentez le gain en chargeant plus de thèses."
