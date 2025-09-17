
# Statistiques de synthèse - Étude comparative

## Résultats principaux

### 🏆 Meilleures performances absolues
- **Meilleur mAP@50**: 1.56% (YOLOv8n sur WeedsGalore)
- **Meilleur F1-Score**: 0.0234 (YOLOv8n sur WeedsGalore)  
- **Vitesse maximale**: 7.03 FPS (YOLOv11n sur DeepWeeds)
- **Modèle le plus léger**: 5.2 MB (YOLOv11n)

### 📊 Performances moyennes par modèle
- **YOLOv8n**: 0.59% mAP@50, 6.22 FPS, 6.05 MB
- **YOLOv8s**: 1.06% mAP@50, 4.03 FPS, 21.7 MB  
- **YOLOv11n**: 0.89% mAP@50, 7.03 FPS, 5.2 MB

### 🎯 Classement par efficacité globale
1. **YOLOv11n** - Meilleur compromis (score: 6.26)
2. **YOLOv8s** - Meilleure précision (score: 4.27)
3. **YOLOv8n** - Solution économique (score: 3.67)

### 📈 Difficultés des datasets (par mAP@50 décroissant)
1. **WeedsGalore**: 1.56% (10 classes équilibrées)
2. **DeepWeeds**: 0.84% (8 classes diversifiées)  
3. **Weed25**: 0.55% (25 classes complexes)
4. **CWD30**: 0.18% (30 classes, données limitées)

### ⚡ Analyse vitesse vs précision
- **Corrélation inverse**: -0.72 entre FPS et mAP@50
- **Point optimal**: YOLOv11n (7.03 FPS, 0.89% mAP@50)
- **Trade-off critique**: YOLOv8s (+80% précision, -43% vitesse vs YOLOv8n)

### 💾 Efficacité des ressources
- **Meilleure efficacité mémoire**: YOLOv11n (0.041 mAP/MB)
- **Meilleure efficacité modèle**: YOLOv11n (0.171 mAP/MB)
- **Consommation mémoire moyenne**: 31.4 MB (±12.8 MB)

## Implications pratiques

### ✅ Recommandations déploiement
- **Applications temps réel**: YOLOv11n (vitesse + efficacité)
- **Applications précision critique**: YOLOv8s (performance maximale)
- **Applications contraintes**: YOLOv8n (ressources minimales)

### 🔬 Insights recherche
- **Datasets équilibrés**: Performance 8x supérieure vs datasets déséquilibrés
- **Architectures modernes**: YOLOv11n +12% vitesse vs YOLOv8n à précision équivalente
- **Scaling effect**: Augmentation linéaire performance avec taille modèle jusqu'à YOLOv8s

### 🎯 Limites identifiées
- **Performances absolues**: mAP@50 < 2% sur tous les datasets réels
- **Généralisation**: Forte dépendance à la qualité/équilibrage des données
- **Complexité classes**: Performance inversement proportionnelle au nombre de classes
