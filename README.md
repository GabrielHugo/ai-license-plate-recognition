# AI License Plate Recognition (ALPR) - Double Stage Detection

Ce projet est un système de reconnaissance automatique de plaques d'immatriculation utilisant l'intelligence artificielle (Computer Vision).

Contrairement aux approches classiques (OCR type Tesseract), ce projet utilise une méthode **"Two-Stage Detection"** (Cascade de détecteurs) pour une robustesse maximale :
1.  **Détection** : Un premier modèle trouve la plaque.
2.  **Lecture** : Un second modèle identifie chaque caractère individuellement.

## Architecture Technique

Le pipeline de traitement se déroule en 3 étapes :

1.  **Localisation (YOLOv8)** : Le modèle `best_plaque.pt` scanne l'image pour trouver la zone de la plaque d'immatriculation.
2.  **Extraction & Crop** : La zone détectée est découpée pour éliminer le bruit visuel environnant.
3.  **Reconnaissance (YOLOv8)** : Le modèle `best_char.pt` analyse le découpage et détecte les caractères (A-Z, 0-9) comme des objets distincts.
4.  **Tri Spatial** : Un algorithme Python trie les coordonnées des caractères de gauche à droite pour reconstituer la chaîne alphanumérique.

## Installation

1.  **Cloner le projet**
    ```bash
    git clone [https://github.com/GabrielHugo/ai-license-plate-recognition.git](https://github.com/GabrielHugo/ai-license-plate-recognition.git)
    cd ai-license-plate-recognition
    ```

2.  **Installer les dépendances**
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install -r requirements.txt
    ```
    *(Ceci installera Ultralytics, OpenCV et NumPy)*

## Guide d'Utilisation

Pour des raisons de confidentialité et de poids, les images de test ne sont pas incluses dans ce dépôt. Voici comment tester le projet avec vos propres données :

1.  **Créer le dossier d'images** :
    Créez un dossier nommé `images` à la racine du projet.

2.  **Ajouter une photo** :
    Déposez une photo de voiture (ex: `voiture_test.jpg`) dans ce dossier.

3.  **Configurer le script** :
    Ouvrez `main.py` et modifiez la variable `image_path` (vers la ligne 15) :
    ```python
    image_path = 'images/voiture_test.jpg'
    ```

4.  **Lancer la détection** :
    ```bash
    python main.py
    ```

Le programme affichera la voiture avec la plaque détectée et le texte lu.

## Contexte Académique

Ce projet a été réalisé dans le cadre de ma formation en programmation d'IA à **Technofutur TIC**. Il met en pratique :
* **Deep Learning** : Entraînement et fine-tuning de modèles YOLOv8.
* **Data Engineering** : Préparation et nettoyage de datasets.
* **Logique Algorithmique** : Traitement et tri des coordonnées spatiales.

### Crédits et Licences

Ce projet a été rendu possible grâce à l'utilisation de datasets open-source pour l'entraînement des modèles IA.

**1. Dataset Plaques (Détection) :**
* **Titre** : *European License Plates Dataset*
* **Auteur** : Patrick Neicu
* **Source** : Roboflow Universe
* **Date** : Mars 2024
* **Lien** : [https://universe.roboflow.com/patrick-neicu/european-license-plates](https://universe.roboflow.com/patrick-neicu/european-license-plates)
* **Licence** : CC BY 4.0

**2. Dataset Caractères (Lecture) :**
* **Titre** : *License Plate Characters Dataset*
* **Auteur** : Adam Toth
* **Source** : Roboflow Universe
* **Date** : August 2023
* **Lien** : [https://universe.roboflow.com/adam-toth-b7suq/license-plate-characters-7qltj](https://universe.roboflow.com/adam-toth-b7suq/license-plate-characters-7qltj)
* **Licence** : CC BY 4.0
