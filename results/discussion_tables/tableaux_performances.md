
## Tableau 1: Performances détaillées par modèle et dataset

| Modèle | Dataset | mAP@50 (%) | F1-Score | Temps inf. (ms) | FPS | Taille (MB) | Mémoire (MB) |
|--------|---------|------------|----------|-----------------|-----|-------------|--------------|
| **YOLOv8n** | DeepWeeds | 0.366 | 0.0026 | 158.5 | 6.31 | 6.0 | 22.9 |
| **YOLOv8s** | DeepWeeds | 1.250 | 0.0154 | 245.2 | 4.08 | 21.5 | 45.7 |
| **YOLOv11n** | DeepWeeds | 0.890 | 0.0087 | 142.3 | 7.03 | 5.2 | 21.8 |
| **YOLOv8n** | Weed25 | 0.240 | 0.0017 | 165.8 | 6.03 | 6.1 | 24.1 |
| **YOLOv8s** | Weed25 | 0.870 | 0.0120 | 251.4 | 3.98 | 21.8 | 47.2 |
| **YOLOv8n** | CWD30 | 0.180 | 0.0013 | 172.1 | 5.81 | 6.2 | 25.3 |
| **YOLOv8n** | WeedsGalore | 1.560 | 0.0234 | 148.7 | 6.73 | 5.9 | 22.1 |


## Tableau 2: Performances moyennes par modèle

| Modèle | mAP@50 (%) | F1-Score | Temps inf. (ms) | FPS | Taille (MB) | Mémoire (MB) |
|--------|------------|----------|-----------------|-----|-------------|--------------|
| **YOLOv11n** | 0.890 | 0.0090 | 142.3 | 7.03 | 5.2 | 21.8 |
| **YOLOv8n** | 0.586 | 0.0070 | 161.3 | 6.22 | 6.1 | 23.6 |
| **YOLOv8s** | 1.060 | 0.0140 | 248.3 | 4.03 | 21.6 | 46.5 |


## Tableau 3: Caractéristiques et complexité des datasets

| Dataset | Nb Classes | Images Train | Images Val | Difficulté (1-5) | mAP@50 Moyen (%) |
|---------|------------|--------------|------------|-------------------|------------------|
| **WeedsGalore** | 10 | 24 | 6 | 2 ⭐⭐ | 1.560 |
| **DeepWeeds** | 8 | 80 | 20 | 3 ⭐⭐⭐ | 0.840 |
| **Weed25** | 25 | 80 | 20 | 4 ⭐⭐⭐⭐ | 0.550 |
| **CWD30** | 30 | 40 | 10 | 5 ⭐⭐⭐⭐⭐ | 0.180 |


## Tableau 4: Métriques d'efficacité

| Modèle | Dataset | Efficacité Mémoire | Efficacité Modèle | Efficacité Vitesse |
|--------|---------|-------------------|-------------------|--------------------|
| **YOLOv8n** | DeepWeeds | 0.0160 | 0.0615 | 2.309 |
| **YOLOv8s** | DeepWeeds | 0.0274 | 0.0581 | 5.100 |
| **YOLOv11n** | DeepWeeds | 0.0408 | 0.1712 | 6.257 |
| **YOLOv8n** | Weed25 | 0.0100 | 0.0392 | 1.447 |
| **YOLOv8s** | Weed25 | 0.0184 | 0.0399 | 3.463 |
| **YOLOv8n** | CWD30 | 0.0071 | 0.0288 | 1.046 |
| **YOLOv8n** | WeedsGalore | 0.0706 | 0.2649 | 10.499 |


## Tableau 5: Recommandations par contexte d'application

| Contexte | Modèle Recommandé | Justification | Configuration Optimale | Performance Attendue |
|----------|-------------------|---------------|------------------------|---------------------|
| **Drone** | YOLOv11n | Meilleur compromis vitesse/efficacité énergétique | Résolution 416x416, INT8 | 12-15 FPS, mAP@50 ~0.75% |
| **Robot Agricole** | YOLOv8s | Précision maximale requise pour traitement localisé | Résolution 640x640, FP16 | 5-8 FPS, mAP@50 >1% |
| **Capteurs Fixes** | YOLOv8n | Efficacité énergétique pour fonctionnement 24/7 | Traitement batch, optimisations | 1-5 min latence, autonomie 6-12 mois |

### Légende des critères d'évaluation:
- **Efficacité Mémoire**: mAP@50 / Mémoire utilisée (MB)
- **Efficacité Modèle**: mAP@50 / Taille du modèle (MB)  
- **Efficacité Vitesse**: mAP@50 × FPS (score composite)
- **Difficulté Dataset**: Basée sur nb classes, taille données, équilibrage
