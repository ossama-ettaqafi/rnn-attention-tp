# **Compte Rendu du TP : Image Captioning avec Attention**

## **I. Introduction**
Ce TP a pour objectif d’implémenter un modèle de *génération automatique de légendes d’images* (image captioning) en combinant :
- Un **encodeur visuel** basé sur ResNet50 pré-entraîné (transfer learning),
- Un **décodeur séquentiel** de type LSTM,
- Un **mécanisme d’attention** permettant au modèle de se concentrer sur différentes régions de l’image lors de la génération de chaque mot.

Le dataset utilisé est **Flickr30k**, contenant 31 783 images et 158 915 légendes.

---

## **II. Structure du projet réalisé**

### **1. Préparation de l’environnement**
- Installation des bibliothèques nécessaires : `torch`, `torchvision`, `transformers`, `tensorboard`, `nltk`.
- Vérification de la disponibilité du GPU (CUDA).

### **2. Chargement et prétraitement des données**
- Lecture du fichier CSV `results.csv`.
- Nettoyage des colonnes et des légendes.
- **Tokenisation** des légendes avec NLTK (`word_tokenize`).
- Construction d’un vocabulaire de **12 509 mots** (fréquence minimale = 2).
- Ajout des tokens spéciaux : `<pad>`, `<sos>`, `<eos>`, `<unk>`.

### **3. Création du Dataset personnalisé**
- Classe `Flickr30kDataset` pour charger images et légendes.
- Transformations appliquées aux images :
  - Redimensionnement à 256x256 → RandomCrop 224x224 → RandomHorizontalFlip → Normalisation (ImageNet).
- Encodage des légendes avec padding (longueur maximale = 30).

### **4. Extraction des caractéristiques visuelles**
- **ResNet50** pré-entraîné, gelé (`requires_grad = False`).
- Extraction des features après la dernière couche convolutionnelle (`avgpool`).
- Format de sortie : `(batch, 49, 2048)` (49 régions spatiales de 2048 dimensions).

### **5. Module d’attention**
- Implémentation de la classe `Attention`.
- Calcul des scores d’attention par combinaison linéaire des features visuelles et de l’état caché du LSTM.
- Softmax pour obtenir les poids → produit avec les features pour obtenir un **vecteur de contexte**.

### **6. LSTM avec mécanisme d’attention**
- Classe `LSTMWithAttention` utilisant `nn.LSTMCell`.
- À chaque pas de temps :
  - Calcul du vecteur de contexte via le module d’attention.
  - Concaténation avec l’embedding du mot courant.
  - Mise à jour des états cachés (h, c).

### **7. Modèle complet**
- Classe `ImageCaptioningModel` intégrant :
  - `FeatureExtractor` (ResNet50)
  - `EmbeddingLayer`
  - `LSTMWithAttention`
- Méthode `generate_caption` pour l’inférence.

### **8. Entraînement**
- **Hyperparamètres** :
  - Taille des embeddings : 256
  - Taille cachée du LSTM : 256
  - Taille de batch : 32
  - Nombre d’époques : 25
  - Optimiseur : Adam (LR = 0.001)
  - Scheduler : StepLR (step=10, gamma=0.5)
  - Fonction de perte : `CrossEntropyLoss` (ignorant `<pad>`)
- **Teacher forcing** : utilisation de la légende réelle comme entrée à chaque pas.
- **Gradient clipping** (max_norm = 1.0) pour éviter l’explosion des gradients.

### **9. Évaluation et résultats**
- **Loss finale** :
  - Train : 3.34
  - Validation : 3.37
- **Exemples de génération** :
  - Le modèle apprend à décrire des scènes simples ("a man in a blue shirt", "group of people sitting").
  - Des répétitions persistent ("blue shirt and a blue shirt...").
  - La sémantique générale est correcte, mais la diversité lexicale est limitée.

### **10. Sauvegarde et chargement**
- Fonctions `save_model` et `load_model` pour persister le modèle entraîné.
- Fichier de checkpoint incluant :
  - Poids du modèle et de l’optimiseur
  - Vocabulaire (`word2idx`, `idx2word`)
  - Hyperparamètres

### **11. Visualisation de l’attention**
- Fonction `visualize_attention` pour extraire les poids d’attention à chaque mot.
- Permet de comprendre sur quelles régions de l’image le modèle se focalise.

---

## **III. Points forts du TP**
✅ **Architecture moderne** : Combinaison ResNet + LSTM + Attention.  
✅ **Gestion efficace du vocabulaire** avec tokens spéciaux et padding.  
✅ **Utilisation de TensorBoard** pour le suivi des courbes de loss.  
✅ **Gradient clipping** et **scheduling du LR** pour stabiliser l’apprentissage.  
✅ **Génération interactive** avec possibilité de visualisation de l’attention.  

---

## **IV. Difficultés rencontrées**
⚠️ **Répétitions dans les légendes générées** → problème classique des modèles séquentiels.  
⚠️ **Temps d’entraînement long** (~37 minutes pour 25 époques sur GPU T4).  
⚠️ **Mémoire GPU limitée** sur Kaggle → réduction de la taille du batch.  
⚠️ **Beam search non implémenté** dans la version finale → génération déterministe (argmax).  

---

## **V. Améliorations possibles**
🔹 **Beam search** pour améliorer la qualité des légendes générées.  
🔹 **Fine-tuning partiel** du ResNet après quelques époques.  
🔹 **Utilisation d’embeddings pré-entraînés** (Word2Vec, GloVe).  
🔹 **Ajout de régularisation** (dropout plus fort, weight decay).  
🔹 **Évaluation quantitative** avec métriques BLEU, METEOR, CIDEr.  
🔹 **Augmentation de données** plus poussée (rotation, changement de couleur).  

---

## **VI. Conclusion**
Ce TP a permis de mettre en œuvre un pipeline complet d’image captioning, depuis le chargement des données jusqu’à la génération de légendes avec visualisation de l’attention. Le modèle appris, bien que perfectible, produit des descriptions cohérentes pour des images simples. Les concepts clés maîtrisés sont :
- Transfer learning avec ResNet
- Mécanismes d’attention
- Gestion de séquences avec LSTM
- Entraînement de modèles multimodaux (image + texte)

**Prochaine étape** : explorer des architectures plus récentes (Transformer, Vision Transformer) et des techniques avancées comme le self-critical sequence training.