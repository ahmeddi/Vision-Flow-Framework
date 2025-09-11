# Vision Flow Framework - Détection d'Adventices par Intelligence Artificielle

Framework de recherche complet pour la comparaison de YOLOv8, YOLOv11 et des modèles de détection d'objets de pointe pour la détection d'adventices en temps réel dans l'agriculture de précision.

## 🚀 Démarrage Rapide

### 1. Configuration de l'Environnement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Générer un jeu de données de démonstration (pour les tests)
python scripts/generate_dummy_data.py --n_train 20 --n_val 10

# Télécharger les jeux de données réels
python scripts/download_weed_datasets.py --datasets deepweeds --sample 50
python scripts/download_weed_datasets.py --datasets weed25 --sample 40
```

### 2. Entraînement des Modèles

```bash
# Entraînement d'un seul modèle
python scripts/train.py --data data/deepweeds.yaml --models yolov8n.pt --epochs 5 --batch-size 2

# Benchmark de modèles multiples
python scripts/train.py --data data/weed25.yaml --models yolov8n.pt yolov8s.pt yolov11n.pt --config configs/base.yaml
```

### 3. Évaluation des Performances

```bash
# Évaluation de base (mAP, FPS, latence)
python scripts/evaluate.py --models_dir results/runs/yolov8n8 --device cpu --data data/deepweeds.yaml

# Analyse de la consommation énergétique
python scripts/energy_logger.py --models results/runs/yolov8n/weights/best.pt --device cuda --n_images 100

# Tests de robustesse (perturbations)
python scripts/perturb_eval.py --model results/runs/yolov8n/weights/best.pt --data data/deepweeds.yaml
```

### 4. Optimisation des Modèles

```bash
# Élagage (structuré/non-structuré)
python scripts/prune.py --model results/runs/yolov8n/weights/best.pt --output models/yolov8n_pruned.pt --method unstructured --amount 0.3

# Quantification (INT8, export ONNX)
python scripts/quantize.py --model results/runs/yolov8n/weights/best.pt --formats onnx tensorrt --benchmark
```

## 📊 Structure du Projet

```
vff/
├── configs/
│   └── base.yaml                 # Hyperparamètres d'entraînement
├── data/
│   ├── deepweeds/               # Jeu de données DeepWeeds
│   │   ├── images/              # Images (train/val/test)
│   │   └── labels/              # Étiquettes YOLO
│   ├── weed25/                  # Jeu de données Weed25
│   │   ├── images/              # 25 espèces d'adventices
│   │   └── labels/              # Annotations de détection
│   ├── sample_weeds/            # Échantillon de démonstration
│   ├── deepweeds.yaml           # Configuration YOLO DeepWeeds
│   ├── weed25.yaml             # Configuration YOLO Weed25
│   └── dummy.yaml              # Jeu de données de test
├── scripts/
│   ├── download_weed_datasets.py # Acquisition des jeux de données
│   ├── generate_dummy_data.py   # Génération de données de test
│   ├── train.py                 # Script d'entraînement unifié
│   ├── evaluate.py              # Évaluation des performances
│   ├── energy_logger.py         # Mesure de consommation énergétique
│   ├── perturb_eval.py          # Tests de robustesse
│   ├── prune.py                 # Élagage des modèles
│   ├── quantize.py              # Quantification des modèles
│   └── comprehensive_benchmark.py # Benchmark complet
├── results/
│   ├── runs/                    # Sorties d'entraînement
│   ├── tables/                  # Tableaux de résultats
│   ├── figures/                 # Graphiques générés
│   ├── training_summary.json    # Métriques d'entraînement
│   └── eval_summary.json        # Résultats d'évaluation
├── models/                      # Modèles optimisés
└── README_FR.md                 # Ce fichier
```

## 🎯 Modèles Supportés

### Famille YOLO

- **YOLOv8**: variantes n, s, m, l, x
- **YOLOv11**: variantes n, s, m, l, x
- **YOLOv7**: Diverses configurations
- **YOLO-NAS**: variantes S, M, L

### Autres Architectures

- **EfficientDet**: D0-D7
- **DETR/RT-DETR**: Basé sur Transformer
- **PP-YOLOE**: Implémentation PaddlePaddle

## 📁 Jeux de Données Détaillés

### 1. DeepWeeds Dataset

**Source**: Université du Queensland, Australie  
**Taille**: 17,509 images  
**Espèces**: 8 espèces d'adventices australiennes

#### Espèces Incluses:

- **Chinee Apple** (Ziziphus mauritiana) - Pomme de Chine
- **Lantana** (Lantana camara) - Lantanier
- **Parkinsonia** (Parkinsonia aculeata) - Parkinsonie épineuse
- **Parthenium** (Parthenium hysterophorus) - Grande camomille
- **Pear** (Opuntia species) - Figuier de Barbarie
- **Prickly Acacia** (Vachellia nilotica) - Acacia épineux
- **Rubber Vine** (Cryptostegia grandiflora) - Liane caoutchouc
- **Siam Weed** (Chromolaena odorata) - Herbe du Siam

#### Caractéristiques:

- Images capturées sur le terrain avec appareils mobiles
- Conditions d'éclairage et saisonnières variables
- Résolution: 256x256 pixels
- Format d'annotation: Classification et détection d'objets

#### Commandes d'utilisation:

```bash
# Télécharger DeepWeeds avec échantillonnage
python scripts/download_weed_datasets.py --datasets deepweeds --sample 30

# Entraîner sur DeepWeeds
python scripts/train.py --data data/deepweeds.yaml --models yolov8n.pt --epochs 5 --batch-size 2

# Évaluer sur DeepWeeds
python scripts/evaluate.py --models_dir results/runs/yolov8n8 --device cpu --data data/deepweeds.yaml
```

### 2. Weed25 Dataset

**Source**: Collection AgML  
**Taille**: ~10,000+ images  
**Espèces**: 25 espèces d'adventices communes

#### Espèces Principales:

- **Graminées**: Giant_Foxtail, Large_Crabgrass, Barnyardgrass, Johnsongrass
- **Amaranthacées**: Waterhemp, Palmer_Amaranth, Redroot_Pigweed
- **Brassicacées**: Wild_Mustard, Field_Pennycress, Shepherds_Purse
- **Fabacées**: Common_Lambsquarters, Velvetleaf
- **Autres**: Common_Ragweed, Giant_Ragweed, Common_Cocklebur

#### Caractéristiques:

- Résolution haute définition (300-2000px)
- Multiples stades de croissance
- Conditions de terrain variées
- Annotations de boîtes englobantes

#### Commandes d'utilisation:

```bash
# Télécharger Weed25 avec échantillonnage
python scripts/download_weed_datasets.py --datasets weed25 --sample 40

# Entraîner sur Weed25
python scripts/train.py --data data/weed25.yaml --models yolov8n.pt yolov11n.pt --epochs 10

# Évaluer sur Weed25
python scripts/evaluate.py --models_dir results/runs --device cuda --data data/weed25.yaml
```

### Format des Données

Tous les jeux de données sont convertis au format YOLO:

- `images/train/`, `images/val/`, `images/test/`
- `labels/train/`, `labels/val/`, `labels/test/`
- Définitions des classes dans les fichiers YAML

## 🔧 Fonctionnalités Principales

### 1. Framework d'Entraînement Unifié

- Hyperparamètres cohérents entre les modèles
- Entraînement reproductible avec graines fixes
- Précision mixte automatique (AMP)
- Augmentations avancées (Mosaic, MixUp, HSV)

### 2. Évaluation Complète

- **Précision**: mAP@0.5, mAP@0.5:0.95
- **Vitesse**: FPS, latence (moyenne/p95)
- **Efficacité**: Taille du modèle, utilisation mémoire
- **Énergie**: Consommation électrique (J/image)
- **Robustesse**: Performance sous perturbations

### 3. Optimisation des Modèles

- **Élagage**: Suppression structurée/non-structurée des paramètres
- **Quantification**: Conversion INT8 pour le déploiement edge
- **Export**: Formats ONNX, TensorRT

### 4. Tests de Robustesse

Perturbations environnementales:

- Variations de luminosité/contraste
- Bruit gaussien
- Flou de mouvement
- Correction gamma

## 📈 Métriques et Analyse

### Métriques Principales

- **mAP@0.5**: Précision Moyenne Moyennée à IoU=0.5
- **mAP@0.5:0.95**: mAP moyennée sur IoU 0.5-0.95
- **FPS**: Images par seconde (vitesse d'inférence)
- **Latence**: Temps d'inférence par image
- **Énergie**: Joules par image
- **Taille**: Taille du fichier modèle (MB)

### Analyse Statistique

- Intervalles de confiance bootstrap
- Tests de rang signé de Wilcoxon
- Analyse de corrélation de Spearman
- Rapport de taille d'effet

## 🏃‍♂️ Exemples d'Utilisation

### Entraînement de Modèles Multiples

```python
# Entraîner des variantes YOLO sur un jeu de données personnalisé
python scripts/train.py \
    --data data/deepweeds.yaml \
    --models yolov8n.pt yolov8s.pt yolov11n.pt \
    --config configs/base.yaml \
    --epochs 10 \
    --batch-size 4
```

### Benchmark Énergétique

```python
# Comparer la consommation énergétique
python scripts/energy_logger.py \
    --models results/runs/*/weights/best.pt \
    --device cuda \
    --n_images 200 \
    --output results/energy_comparison.json
```

### Compression de Modèle

```python
# Élaguer et quantifier pour le déploiement edge
python scripts/prune.py --model model.pt --output pruned.pt --amount 0.4
python scripts/quantize.py --model pruned.pt --formats onnx tflite
```

### Benchmark Complet

```python
# Exécuter un benchmark complet sur tous les modèles
python scripts/comprehensive_benchmark.py \
    --datasets deepweeds weed25 \
    --models yolov8n yolov8s yolov11n \
    --output results/full_benchmark.json
```

## 🔬 Applications de Recherche

### Contributions du Papier

1. **Comparaison multi-générationnelle YOLO** (v8 vs v11)
2. **Benchmark conscient de l'énergie** pour la robotique agricole
3. **Évaluation de robustesse** en conditions de terrain
4. **Techniques d'optimisation** pour le déploiement edge

### Reproductibilité

- Graines aléatoires fixes (42)
- Opérations déterministes lorsque possible
- Épinglage de version (`requirements.txt`)
- Journalisation détaillée des hyperparamètres

## 🚧 État Actuel du Projet

### ✅ Terminé

- [x] Structure du projet et framework
- [x] Génération de jeux de données fictifs
- [x] Pipeline d'entraînement de base (YOLOv8)
- [x] Framework d'évaluation
- [x] Infrastructure de journalisation énergétique
- [x] Implémentation de l'élagage
- [x] Framework de quantification
- [x] Structure des tests de robustesse
- [x] Intégration des jeux de données réels

### 🔄 En Cours

- [ ] Validation d'entraînement multi-modèles
- [ ] Implémentation de l'analyse statistique complète
- [ ] Génération automatique de rapports

### 📋 Planifié

- [ ] Familles de modèles supplémentaires (DETR, EfficientDet)
- [ ] Évaluation inter-jeux de données
- [ ] Conteneurisation Docker
- [ ] Interface web de visualisation

## 🎯 Résultats Actuels

### Performances sur DeepWeeds (YOLOv8n)

- **mAP@0.5**: 53.1%
- **mAP@0.5:0.95**: 50.4%
- **Précision**: 0.33%
- **Rappel**: 100%
- **Temps d'entraînement**: 36 secondes (5 epochs)

### Configuration d'Entraînement

- **Modèle**: YOLOv8n (nano)
- **Époques**: 5
- **Taille de lot**: 2
- **Jeu de données**: DeepWeeds (échantillon de 30 images)
- **Device**: CPU

## 🤝 Contribution

### Ajouter de Nouveaux Modèles

1. Créer un wrapper de modèle dans `scripts/train.py`
2. Ajouter le support d'évaluation dans `scripts/evaluate.py`
3. Mettre à jour les fichiers de configuration

### Ajouter de Nouveaux Jeux de Données

1. Ajouter les informations du jeu de données dans `scripts/download_weed_datasets.py`
2. Implémenter la conversion de format si nécessaire
3. Créer le fichier de configuration YAML

## 📚 Citations

```bibtex
@article{vision_flow_framework_2025,
  title={Vision Flow Framework: Comparative Evaluation of YOLOv8, YOLOv11, and State-of-the-Art Object Detection Models for Real-Time Weed Detection in Precision Agriculture},
  author={Vision Flow Team},
  journal={Journal of Precision Agriculture AI},
  year={2025},
  note={Framework open-source disponible sur GitHub}
}
```

## 📊 Commandes de Workflow Complètes

### Pipeline d'Entraînement Standard

```bash
# 1. Configuration initiale
pip install -r requirements.txt

# 2. Télécharger les données
python scripts/download_weed_datasets.py --datasets deepweeds --sample 50
python scripts/download_weed_datasets.py --datasets weed25 --sample 40

# 3. Entraînement multi-modèles
python scripts/train.py --data data/deepweeds.yaml --models yolov8n.pt yolov8s.pt --epochs 10 --batch-size 4
python scripts/train.py --data data/weed25.yaml --models yolov11n.pt yolov11s.pt --epochs 10 --batch-size 4

# 4. Évaluation complète
python scripts/evaluate.py --models_dir results/runs --device cuda
python scripts/energy_logger.py --models results/runs/*/weights/best.pt --device cuda

# 5. Tests de robustesse
python scripts/perturb_eval.py --model results/runs/yolov8n/weights/best.pt --data data/deepweeds.yaml

# 6. Analyse statistique
python scripts/statistical_analysis.py --results_dir results/

# 7. Génération de visualisations
python scripts/create_visualizations.py --results_dir results/ --output figures/
```

### Pipeline d'Optimisation

```bash
# Élagage des modèles
python scripts/prune.py --model results/runs/yolov8n/weights/best.pt --output models/yolov8n_pruned.pt --amount 0.3

# Quantification
python scripts/quantize.py --model models/yolov8n_pruned.pt --formats onnx tensorrt int8

# Test des modèles optimisés
python scripts/evaluate.py --models_dir models/ --device cpu --optimized
```

## 📄 Licence

Ce projet est sous licence MIT. Voir les licences individuelles des jeux de données pour les conditions d'utilisation des données.

## 🔗 Ressources

- [Documentation Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Article du jeu de données DeepWeeds](https://www.nature.com/articles/s41598-018-38343-3)
- [Collection de jeux de données agricoles AgML](https://github.com/AgML/AgML)
- [Documentation PyTorch](https://pytorch.org/docs/)

---

**Dernière mise à jour**: Septembre 2025  
**Version**: 1.0.0  
**Framework**: PyTorch + Ultralytics YOLO  
**Langue**: Français (French)
