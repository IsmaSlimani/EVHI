Nous avons developpé sur unity notre propre environnement virtuel interactif, un jeu de combat où le joueur est connecté à des capteurs d'activité musculaire (EMG) afin d'effectuer des actions dans le jeu.

Le code de ce repository contient les modèles crées pour la reconnaissances de gestes et la détéction du niveau de fatigue.

## Etapes principales 

1. Collecte et prétraitement des données à partir de signaux EMG.
2. Extraction de caractéristiques pertinentes des signaux.
3. Implémentation d'un modèle de classification supervisée
HistGradientBoostingClassifier) pour la détection des
mouvements.
4. Clustering des données avec KMeans pour labelliser les
niveaux de fatigue musculaire.
5. Evaluation des modèles.
