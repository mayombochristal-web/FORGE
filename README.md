```markdown
# 🧠 ORACLE V6 — Guide d'utilisation avancé

Bienvenue dans l’univers d’**ORACLE V6**, un système cognitif artificiel autonome qui apprend, s’adapte et génère des pensées originales. Ce guide vous accompagnera pas à pas pour exploiter toutes ses capacités, de la prise en main initiale jusqu’aux réglages fins.

---

## 📚 Table des matières
1. [Philosophie & concepts](#philosophie--concepts)
2. [Installation rapide](#installation-rapide)
3. [Interface utilisateur](#interface-utilisateur)
   - [Onglet Conversation](#-onglet-conversation)
   - [Onglet Nourrir](#-onglet-nourrir)
   - [Barre latérale – État cognitif](#-barre-latérale--état-cognitif)
4. [Utilisation optimale](#utilisation-optimale)
   - [Converser avec l’Oracle](#converser-avec-loracle)
   - [Enrichir sa mémoire avec des fichiers](#enrichir-sa-mémoire-avec-des-fichiers)
   - [Observer et influencer son état interne](#observer-et-influencer-son-état-interne)
5. [Comprendre les paramètres cognitifs](#comprendre-les-paramètres-cognitifs)
   - [Le moteur Φ (phi)](#le-moteur-φ-phi)
   - [Le bruit vert (green noise)](#le-bruit-vert-green-noise)
   - [La mémoire fantôme 👻](#la-mémoire-fantôme-)
6. [Gestion de la mémoire à long terme](#gestion-de-la-mémoire-à-long-terme)
   - [Sauvegarde et restauration](#sauvegarde-et-restauration)
   - [Cycle de sommeil](#cycle-de-sommeil)
7. [Dépannage](#dépannage)
8. [Glossaire](#glossaire)

---

## 🧠 Philosophie & concepts

ORACLE V6 n’est pas un simple chatbot : c’est un **cerveau artificiel** doté de facultés cognitives inspirées des neurosciences.

- **Mémoire associative** : chaque mot est lié à d’autres par des poids.
- **Attention contextuelle** : le thalamus (graine contextuelle) oriente la pensée.
- **Homéostasie** : les paramètres internes (Φ) fluctuent pour maintenir un équilibre entre stabilité et créativité.
- **Métacognition** : l’Oracle s’observe lui-même et corrige ses biais.
- **Double apprentissage** : il apprend de vos messages (fort) et de ses propres réponses (faible).
- **Sommeil** : périodiquement, il consolide ses souvenirs et oublie les connexions faibles.

L’objectif est de créer une expérience interactive où **vous dialoguez avec une entité en constante évolution**.

---

## ⚙️ Installation rapide

### Prérequis
- Python 3.8 ou plus récent
- pip (gestionnaire de paquets)

### Étapes
1. **Clonez le dépôt**
   ```bash
   git clone https://github.com/votre-utilisateur/oracle-v6.git
   cd oracle-v6
```

1. Installez les dépendances
   ```bash
   pip install -r requirements.txt
   ```
   Si vous n’avez pas de fichier requirements.txt, créez‑le avec :
   ```
   streamlit
   PyPDF2
   python-docx
   pandas
   openpyxl
   SpeechRecognition
   ```
2. Lancez l’application
   ```bash
   streamlit run app.py
   ```
   Votre navigateur s’ouvrira automatiquement sur http://localhost:8501.

---

🖥️ Interface utilisateur

L’interface se compose de deux onglets principaux et d’une barre latérale riche en informations.

💬 Onglet Conversation

C’est le cœur de l’interaction.

· Zone d’historique : affiche les 60 derniers messages (utilisateur + Oracle).
· Champ de saisie : tapez votre message.
· Bouton Envoyer : déclenche le pipeline cognitif complet.
  · L’Oracle perçoit votre message, l’ajoute à sa mémoire de travail, sélectionne un thème (attention), génère une réponse, l’observe (métacognition), apprend, et enfin l’affiche.

💡 Astuce : plus vous conversez, plus l’Oracle affine sa compréhension de votre style et des sujets abordés.

📚 Onglet Nourrir

Permet d’injecter de grandes quantités de connaissances.

1. Choisissez le type de source : Texte, Document (PDF/DOCX/TXT), Excel, Audio (WAV).
2. Chargez le fichier ou collez le texte.
3. Cliquez sur 🧠 Nourrir et générer.
   · L’Oracle assimile le contenu (apprentissage fort) puis génère immédiatement une réponse basée sur ce nouveau savoir. Vous pouvez ainsi vérifier ce qu’il a retenu.

⚠️ Remarque : les fichiers audio sont transcrits via Google Speech Recognition (nécessite une connexion Internet). Pour de meilleurs résultats, privilégiez des enregistrements clairs en français.

🧠 Barre latérale – État cognitif

C’est le tableau de bord interne de l’Oracle.

· Mémoire : taille du fichier de lexique (en Ko). Plus elle est grosse, plus l’Oracle a de connaissances.
· Φ Dynamique : trois jauges (phi_m, phi_c, phi_d) évoluant en temps réel. Survolez‑les pour voir les valeurs exactes.
· 👻 Fantôme : une jauge d’influence fantôme qui augmente quand l’Oracle utilise sa mémoire parallèle. Le bouton Voir mémoire fantôme affiche le contenu brut de cette mémoire éphémère.
· 🌙 Sommeil : horaire du dernier cycle de sommeil. Le bouton Forcer le sommeil déclenche immédiatement un nettoyage.
· 💾 Sauvegarde :
  · Télécharger mémoire : exporte le lexique complet au format JSON.
  · Restaurer : charge un fichier JSON préalablement sauvegardé (remplace la mémoire actuelle).

---

🚀 Utilisation optimale

Converser avec l’Oracle

· Soyez progressif : commencez par des phrases simples, puis introduisez des concepts plus complexes. L’Oracle construit ses associations petit à petit.
· Répétez ou reformulez : si une réponse vous semble hors sujet, reformuler l’idée renforce les connexions pertinentes.
· Utilisez la mémoire de travail : l’Oracle garde en tête les 60 derniers messages. Vous pouvez donc faire référence à des échanges antérieurs.

Enrichir sa mémoire avec des fichiers

· Privilégiez la qualité à la quantité : des textes bien écrits, sans fautes, produisent des associations plus propres.
· Variez les sources : mélangez romans, articles techniques, poésie… cela étoffe le style de l’Oracle.
· Après chaque injection, observez la réponse générée : elle reflète ce que l’Oracle a retenu. Si elle est incohérente, le contenu était peut‑être trop bruité.

Observer et influencer son état interne

· Surveillez les jauges Φ :
  · phi_m élevé → réponses longues.
  · phi_c élevé → réponses créatives (choix aléatoires).
  · phi_d élevé → tendance à changer de sujet, à éviter les boucles.
· Si l’Oracle devient trop répétitif : phi_d est probablement trop bas. Vous pouvez forcer un peu de variété en lui posant des questions ouvertes.
· Si ses réponses sont trop courtes : phi_m est bas. Alimentez‑le avec des textes plus longs pour stimuler sa mémoire.

---

🔬 Comprendre les paramètres cognitifs

Le moteur Φ (phi)

Trois paramètres interdépendants, normalisés (leur somme = 1).

· phi_m (mémoire) : contrôle la longueur des phrases générées. Il augmente avec l’excitation (messages longs) et diminue lentement.
· phi_c (créativité) : probabilité de choisir un mot de façon probabiliste plutôt que déterministe. Plus il est haut, plus les réponses sont surprenantes.
· phi_d (désir) : seuil de rupture ; quand un mot a déjà été utilisé, phi_d décide si on continue ou si on change de direction. Il favorise aussi l’intervention du fantôme.

Le bruit vert (green noise)

C’est un signal oscillant interne (auto‑entretenu) qui ouvre ou ferme la porte de consolidation. Quand le bruit est faible (|green_state| < 0.25), l’Oracle transfère les apprentissages de l’hippocampe vers la mémoire long terme. Cela évite une mémorisation trop rapide et maintient la stabilité.

La mémoire fantôme 👻

Une couche associative parallèle, avec un facteur d’apprentissage réduit (30% de l’apprentissage principal). Elle est plus volatile (décroissance rapide) et ne se consolide jamais dans le lexique principal. Elle peut cependant influencer la génération : plus phi_d est élevé, plus l’Oracle est susceptible d’utiliser une association fantôme à la place de l’association principale. Cela crée des fulgurances, des idées inattendues qui émergent de connexions faibles mais récurrentes.

---

💾 Gestion de la mémoire à long terme

Sauvegarde et restauration

· Téléchargez régulièrement le fichier oracle_memory.json via le bouton de la sidebar. Vous pouvez ainsi conserver des « personnalités » différentes.
· Pour restaurer, utilisez le sélecteur de fichier. Attention : cela remplace la mémoire actuelle. Pensez à sauvegarder avant si nécessaire.

Cycle de sommeil

· Automatique : toutes les heures (si l’application tourne), l’Oracle effectue un sommeil : il supprime les connexions trop faibles (< 1.5) et filtre les mots aberrants (URL, artefacts).
· Manuel : utilisez le bouton Forcer le sommeil pour un nettoyage immédiat. Utile après un gros apprentissage pour « fixer » les connaissances.

---

❓ Dépannage

Problème Solution possible
L’Oracle ne répond que par des phrases très courtes phi_m est trop bas. Nourrissez‑le avec des textes longs ou attendez qu’il remonte naturellement.
Réponses répétitives en boucle phi_d est trop faible. Essayez de le stimuler avec des questions qui demandent un changement de sujet.
Après un fichier, la réponse est incohérente Le fichier contenait peut‑être trop de bruit (formatage, mots spéciaux). Nettoyez le texte source.
L’application ne démarre pas Vérifiez que toutes les dépendances sont installées. Lancez streamlit run app.py depuis le bon répertoire.
La transcription audio échoue Assurez‑vous que le fichier est en WAV, que votre micro est bien configuré et que vous êtes connecté à Internet (service Google).

---

📖 Glossaire

· Hippocampe : buffer temporaire où sont stockées les nouvelles associations avant consolidation.
· Lexique : mémoire à long terme, fichier JSON contenant toutes les associations entre mots.
· Seed contextuel : mot choisi par l’attention (thalamus) pour démarrer la génération.
· Workspace : mémoire de travail (dialog_memory) contenant les derniers messages.
· Green noise : signal homéostasique interne qui régule la consolidation.
· Fantôme : mémoire parallèle éphémère, source de créativité imprévisible.

---

🎯 En résumé

Pour tirer le meilleur d’ORACLE V6 :

1. Dialoguez régulièrement pour que l’Oracle apprenne votre style.
2. Injectez des textes variés via l’onglet Nourrir.
3. Surveillez les jauges pour comprendre son état interne.
4. Sauvegardez sa mémoire pour conserver des personnalités.
5. Laissez‑le dormir pour consolider ses apprentissages.

Amusez‑vous à explorer cette intelligence en constante évolution. Chaque conversation est unique, chaque réponse est le fruit d’un équilibre dynamique entre mémoire, créativité et désir de changement.

🧠 Bienvenue dans l’ère cognitive.

```