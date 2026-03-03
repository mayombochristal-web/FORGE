# test_oracle.py
# Script de test pour Oracle Brain V6.5 (deep learning + masque + connecteurs)

import time
from oracle_core import OracleBrain

def test_oracle():
    print("🧪 TEST DE L'ORACLE BRAIN V6.5")
    print("="*50)

    # 1. Initialisation avec un fichier mémoire temporaire
    brain = OracleBrain("test_memory.json")
    print("✅ Cerveau initialisé avec test_memory.json")

    # 2. Apprentissage de quelques phrases
    print("\n📚 Phase d'apprentissage...")
    phrases = [
        "Le code Python est élégant et puissant.",
        "Les algorithmes de tri sont fondamentaux en informatique.",
        "La programmation fonctionnelle utilise des fonctions pures.",
        "L'intelligence artificielle repose sur les réseaux de neurones.",
        "Les bases de données relationnelles stockent des informations."
    ]
    for p in phrases:
        brain.process_input(p)
        print(f"   Appris : {p}")
        time.sleep(0.2)

    print("✅ Apprentissage terminé.")

    # 3. Test de génération avec une question simple
    print("\n💬 Test de conversation :")
    question = "Parle moi du code Python"
    print(f"   User : {question}")
    reponse = brain.process_input(question)
    print(f"   Oracle : {reponse}")

    # 4. Vérification de l'injection de connecteurs
    print("\n🔗 Test de l'injecteur de connecteurs :")
    # On force une réponse courte en vidant la mémoire de dialogue (simule un reset)
    brain.dialog_memory.clear()
    # On donne une entrée très simple pour voir si des connecteurs apparaissent
    reponse_courte = brain.process_input("Bonjour")
    print(f"   User : Bonjour")
    print(f"   Oracle : {reponse_courte}")
    # Vérifie si un connecteur est présent (au début ou au milieu)
    connecteurs = ["cependant", "néanmoins", "d'autre part", "en revanche",
                   "par ailleurs", "toutefois", "ainsi", "donc", "par conséquent"]
    if any(c in reponse_courte.lower() for c in connecteurs):
        print("   ✅ Connecteur détecté !")
    else:
        print("   ⚠️ Aucun connecteur visible (peut-être normal selon le contexte).")

    # 5. Test du Nexus (croisement de deux mémoires)
    print("\n🌐 Test du Nexus (croisement de fichiers) :")
    # Créons un second fichier mémoire avec un autre contenu
    brain2 = OracleBrain("test_nexus.json")
    phrases2 = [
        "L'architecture des microservices est modulaire.",
        "Les conteneurs Docker simplifient le déploiement.",
        "Kubernetes orchestre les conteneurs.",
        "Le cloud computing offre une scalabilité horizontale."
    ]
    for p in phrases2:
        brain2.process_input(p)
    # Maintenant, on croise la mémoire principale avec la seconde
    brain.cross_reference("test_nexus.json")
    print("   ✅ Nexus activé entre test_memory.json et test_nexus.json")
    # On pose une question qui pourrait bénéficier du Nexus
    question_nexus = "Parle moi de déploiement et de cloud"
    print(f"   User : {question_nexus}")
    reponse_nexus = brain.process_input(question_nexus)
    print(f"   Oracle : {reponse_nexus}")

    # 6. Test de la consolidation et du sommeil
    print("\n💤 Test du cycle de sommeil (consolidation et élagage) :")
    # Afficher la taille du lexique avant sommeil
    taille_avant = sum(len(v) for v in brain.lexicon.values())
    print(f"   Nombre de transitions avant sommeil : {taille_avant}")
    brain.sleep_cycle()
    taille_apres = sum(len(v) for v in brain.lexicon.values())
    print(f"   Nombre de transitions après sommeil  : {taille_apres}")
    if taille_apres <= taille_avant:
        print("   ✅ Élagage effectué (taille réduite ou stable).")
    else:
        print("   ⚠️ La taille a augmenté (comportement inattendu).")

    # 7. Test de persistance (rechargement du fichier)
    print("\n💾 Test de persistance (rechargement depuis le fichier) :")
    brain3 = OracleBrain("test_memory.json")
    # Vérifier que le lexique n'est pas vide
    if brain3.lexicon:
        print("   ✅ Mémoire rechargée avec succès.")
    else:
        print("   ❌ Échec du rechargement.")

    print("\n🎉 Tous les tests sont terminés. Vérifiez visuellement les réponses pour vous assurer qu'elles sont cohérentes et contiennent parfois des connecteurs.")

if __name__ == "__main__":
    test_oracle()