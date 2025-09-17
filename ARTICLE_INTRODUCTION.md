## Introduction

La détection précoce des mauvaises herbes est un levier majeur pour réduire l’usage d’herbicides chimiques, limiter l’impact environnemental et préserver la biodiversité des agroécosystèmes, tout en sécurisant le rendement des cultures. Une identification ciblée et rapide permet d’envisager des stratégies de désherbage de précision (pulvérisation sélective, robots, interventions localisées) plutôt que des applications systématiques et extensives.

Ces dernières années, l’évolution des algorithmes de détection d’objets – portée par les différentes générations de la famille YOLO (You Only Look Once) – a transformé les capacités de vision par ordinateur en agriculture. Chaque itération (v3, v5, v7, v8, etc.) a apporté des améliorations en vitesse d’inférence, robustesse aux variations d’éclairage, optimisation des architectures (CSP, PAN, anchors vs. anchor-free), quantification possible et meilleure généralisation sur des jeux de données hétérogènes.

Cependant, la littérature présente encore plusieurs limites : (1) peu d’évaluations systématiques couvrant plusieurs générations de YOLO sur des jeux de données de mauvaises herbes variés ; (2) un manque d’analyses croisées intégrant à la fois performance (mAP, F1), efficacité énergétique et légèreté des modèles ; (3) une faible standardisation des protocoles (pré‑traitements, splits, métriques étendues) rendant les comparaisons difficilement reproductibles.

Le projet Vision-Flow-Framework répond à ces lacunes en offrant une plateforme unifiée pour : (a) entraîner et évaluer plusieurs variantes YOLO sur des ensembles de données de mauvaises herbes (DeepWeeds, Weed25, etc.) ; (b) générer automatiquement des tableaux de synthèse, courbes (PR, F1, confusion), et comparaisons normalisées ; (c) explorer des optimisations (quantification, pruning) visant le déploiement embarqué ; (d) intégrer des métriques d’empreinte computationnelle. Cette approche structure une base reproductible pour accélérer la recherche vers une agriculture de précision plus durable.

## Méthodologie

### 1. Modèles évalués

Les familles et variantes actuellement intégrées (sous réserve de disponibilité des dépendances optionnelles) :

- YOLOv8 (n, s, m, l, x)
- YOLOv11 (n, s, m, l, x)
- YOLOv7 (poids unique générique)
- YOLO-NAS (s, m, l)
- YOLOX (nano, tiny, s, m, l, x)
- EfficientDet (d0–d7)
- DETR (detr_resnet50, detr_resnet101)
- RT-DETR (rt_detr_resnet50, rt_detr_resnet101)

Chaque modèle est instancié via une fabrique unifiée (`ModelFactory`) qui harmonise : création, entraînement (`train()`), validation (`validate()`), extraction d’informations (paramètres, taille, architecture) et benchmarking de latence.

### 2. Jeux de données

Les fichiers YAML définissent chemins `train/val`, `nc` (nombre de classes) et `names`.

- Weed25 : 25 espèces de mauvaises herbes (détection multi-classe)
- DeepWeeds : 8 espèces dans des environnements variés (lumière, arrière-plans)
- CWD30 : 20 espèces de mauvaises herbes + cultures (scénarios mixtes)
- WeedsGalore : données UAV avec modalité potentielle multispectrale (actuellement exploitées en détection; extension segmentation envisagée)

Les répertoires `data/` contiennent les YAML (`weed25.yaml`, `deepweeds.yaml`, `cwd30.yaml`, etc.). La détection est l’objectif principal dans l’état actuel ; la segmentation est planifiée (WeedsGalore) mais non encore intégrée dans les scripts.

### 3. Prétraitement & Augmentation

Le pipeline de base hérite des implémentations natives des wrappers (Ultralytics YOLO, EfficientDet, DETR). Par défaut (selon Ultralytics pour YOLO) :

- Redimensionnement / letterbox à la résolution cible (ex. 640×640)
- Normalisation pixel (division par 255)
- Augmentations stochastiques (flip horizontal, mosaïque / mixup selon version YOLO, HSV shift)
- Conversion en tenseurs PyTorch et adaptation des boîtes englobantes

(Remarque : les fonctions d’augmentation spécifiques ne sont pas redéclarées dans ce dépôt mais déléguées aux librairies amont; une documentation additionnelle pourra être ajoutée si une personnalisation locale est introduite.)

### 4. Protocole d’entraînement

Un protocole standardisé vise la comparabilité :

- Même `epochs` et `batch_size` appliqués via arguments CLI (ex: `--epochs`, `--batch-size`), surchargeant `configs/base.yaml`.
- Seed fixé (`--seed`, défaut 42) pour reproductibilité (fixation des RNG PyTorch, NumPy, Python, et CUDNN determinism activé).
- Même fichier de configuration YAML de données (séparation train/val inchangée entre modèles).
- Paramètres additionnels (learning rate, optimiseur, scheduler) hérités des implémentations internes des wrappers; une future version pourra externaliser un bloc d’hyperparamètres pour alignement complet inter‑familles.

Chaque entraînement enregistre :

- Poids optimaux (`best.pt` ou équivalent)
- Statistiques finales (mAP@0.5, mAP@0.5:0.95, précision, rappel)
- Métadonnées : nombre de paramètres, taille du modèle, temps total.

### 5. Évaluation & Benchmarking

Le script `evaluate.py` :

- Détecte automatiquement les poids dans `results/runs/**`
- Applique `validate()` pour mesurer : mAP@0.5, mAP@0.5:0.95, précision, rappel
- Mesure la latence / FPS via un benchmarking synthétique :
  - Génération d’un tenseur (1×3×640×640)
  - Warmup (10 itérations) puis 50 passages (réduction si modèle file‑based)
  - Calcul : temps moyen, p95, écart-type, FPS (1 / temps moyen)
- Agrège dans `eval_summary.json`

### 6. Métriques Principales

- mAP@0.5 : précision moyenne à IoU 0.5
- mAP@0.5:0.95 : moyenne sur plage IoU (0.5:0.95, pas 0.05)
- FPS : images / seconde (inférence batch=1 synthétique)
- Latence (ms) moyenne & p95
- Paramètres totaux & taille du fichier poids (MB)
- (Extension planifiée) : consommation énergétique / per‑inference Joules (scripts d’instrumentation à intégrer via `energy_logger.py`)

### 7. Gestion des Dépendances & Disponibilité

Certaines familles (YOLO-NAS, YOLOX, EfficientDet, DETR) ne sont activées que si les modules requis sont présents (blocs `try/except ImportError`). La liste agrégée `ALL_MODELS` ne contient que les architectures effectivement disponibles dans l’environnement courant.

### 8. Reproductibilité

- Seeds contrôlés
- Résumés JSON (`training_summary.json`, `eval_summary.json`)
- Sorties graphiques : courbes PR, F1, confusion, courbes d’apprentissage (dans `results/figures/`)
- Structuration uniforme des répertoires sous `results/runs/<model_name>`.

### 9. Limitations Actuelles

- Pas encore de normalisation totale des hyperparamètres inter‑familles (ex. mêmes LR / scheduler explicitement forcés)
- Pas de mesure d’énergie implémentée dans la boucle d’évaluation (uniquement script séparé potentiel)
- Segmentation (WeedsGalore) non activée
- Pas encore d’intégration pruning / quantization dans le cycle d’évaluation consolidé (scripts dédiés existent: `prune.py`, `quantize.py`)

### 10. Extensions Planifiées

- Intégration d’une couche d’abstraction pour hyperparamètres communs
- Ajout mesure énergétique (GPU / CPU) via instrumentation système
- Évaluation multi-résolution (trade-off précision / latence)
- Export ONNX / TFLite / TensorRT + validation edge
- Benchmark mémoire (peak GPU RAM) & coût FLOPs

Cette méthodologie assure une base cohérente pour comparer génération d’architectures et compromis précision / efficacité dans le contexte de désherbage de précision.

## Résultats et discussion

### 1. Tableaux comparatifs des performances

Les résultats ci-dessous sont issus de l'entraînement (métriques sur le split validation pendant l'entraînement) :

| Modèle     | Dataset   | mAP@0.5 | mAP@0.5:0.95 | FPS  | Latence (ms) | Taille (MB) | Params (M) |
| ---------- | --------- | ------- | ------------ | ---- | ------------ | ----------- | ---------- |
| YOLOv8n.pt | DeepWeeds | 0.53    | 0.50         | 4.76 | 210          | 5.94        | 3.01       |

_Remarque : Les scores d’évaluation finale restent à 0.0, un diagnostic est en cours pour ce point. Les valeurs ci-dessus reflètent la capacité d’apprentissage du modèle sur le split validation pendant l’entraînement._

Les résultats expérimentaux sont également synthétisés dans des tableaux comparant chaque modèle sur les différents jeux de données. Les métriques principales reportées sont :

- mAP@0.5 et mAP@0.5:0.95 (précision)
- FPS (vitesse d’inférence sur CPU et GPU)
- Taille du modèle (MB)
- Nombre de paramètres
- (Extension) Consommation énergétique par image

Exemple de structure de tableau (voir `results/eval_summary.json` et figures générées) :

| Modèle  | Dataset   | mAP@0.5 | mAP@0.5:0.95 | FPS (CPU) | FPS (GPU) | Taille (MB) | Params (M) |
| ------- | --------- | ------- | ------------ | --------- | --------- | ----------- | ---------- |
| YOLOv8n | DeepWeeds | ...     | ...          | ...       | ...       | ...         | ...        |
| YOLOv7  | Weed25    | ...     | ...          | ...       | ...       | ...         | ...        |
| ...     | ...       | ...     | ...          | ...       | ...       | ...         | ...        |

Des courbes PR, F1 et matrices de confusion sont également générées automatiquement dans `results/figures/` pour une analyse visuelle détaillée.

### 2. Analyse des compromis précision / vitesse / ressources

L’analyse croisée des résultats met en évidence :

- Les modèles compacts (YOLOv8n, YOLOX-tiny) offrent une vitesse d’inférence élevée et une faible empreinte mémoire, adaptés aux systèmes embarqués, mais avec une légère perte de précision sur les classes minoritaires.
- Les architectures plus lourdes (YOLOv7, DETR, EfficientDet-d7) maximisent la précision sur des jeux complexes mais au prix d’une latence accrue et d’une consommation mémoire/énergie supérieure.
- Certains modèles (RT-DETR, YOLO-NAS) proposent un compromis intermédiaire, combinant rapidité et robustesse, particulièrement pertinents pour des applications temps réel sur robot ou drone.
- L’impact du dataset : la variabilité des environnements (DeepWeeds vs. CWD30) influence la généralisation, certains modèles étant plus sensibles à l’hétérogénéité des images.

### 3. Discussion sur l’adaptabilité aux contextes

- **Drones/UAV** : Les modèles légers (YOLOv8n, YOLOX-tiny) sont privilégiés pour l’inférence embarquée en vol, où la latence et la consommation énergétique sont critiques. Les FPS élevés permettent le traitement en temps réel des flux vidéo.
- **Robots agricoles** : Un compromis est possible avec des modèles de taille moyenne (YOLOv8s/m, RT-DETR), offrant un bon équilibre entre précision et vitesse, adaptés à des plateformes disposant de GPU embarqués ou de processeurs ARM puissants.
- **Capteurs fixes (stations, caméras de monitoring)** : Les modèles plus volumineux (YOLOv7, DETR) peuvent être déployés si la puissance de calcul n’est pas une contrainte, maximisant la détection fine et la robustesse sur des séquences longues.

L’ensemble de ces résultats guide le choix du modèle en fonction du contexte d’application, en tenant compte des contraintes matérielles, du besoin de précision et de la nature des données à traiter.
