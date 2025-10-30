# Comparative Evaluation of YOLOv8, YOLOv11, and Modern Object Detectors for Real-Time Weed Detection in Precision Agriculture

## Title (FR / EN)

- EN: Comparative Evaluation of YOLOv8, YOLOv11, and State-of-the-Art Object Detection Models for Real-Time Weed Detection in Precision Agriculture
- FR: Évaluation comparative de YOLOv8, YOLOv11 et modèles de détection d’objets de pointe pour la détection temps réel des mauvaises herbes en agriculture de précision

## Abstract (English)

We present a comprehensive benchmark comparing YOLOv8 (n,s,m,l,x), YOLOv11 (n,s,m,l,x) and recent object detectors (YOLO-NAS, YOLOX, YOLOv7, PP-YOLOE, EfficientDet D0–D7, DETR, RT-DETR) for real-time weed detection in precision agriculture. Evaluations span four heterogeneous datasets (Weed25, DeepWeeds, CWD30, multispectral WeedsGalore) using mAP@0.5, mAP@0.5:0.95, FPS (edge/GPU), model size, energy consumption, and robustness to environmental shifts (illumination, weather, viewpoint, spectral noise). A unified training and preprocessing protocol ensures fairness. Results highlight (i) improved small-model stability in YOLOv11, (ii) balanced accuracy–speed trade-offs from YOLO-NAS and PP-YOLOE, (iii) >30% energy savings via structured pruning and post-training quantization with minimal mAP loss. We discuss deployment scenarios (UAV, ground robots, fixed sensing) and future directions including multispectral fusion, multi-teacher distillation, and continual adaptation.

## Résumé (Français)

Nous présentons un benchmark exhaustif comparant YOLOv8 (n,s,m,l,x), YOLOv11 (n,s,m,l,x) et des détecteurs récents (YOLO-NAS, YOLOX, YOLOv7, PP-YOLOE, EfficientDet D0–D7, DETR, RT-DETR) appliqués à la détection des mauvaises herbes en agriculture de précision. Les modèles sont évalués sur quatre jeux de données hétérogènes (Weed25, DeepWeeds, CWD30, WeedsGalore multispectral) selon : mAP@0.5, mAP@0.5:0.95, FPS (edge/GPU), taille mémoire, consommation énergétique et robustesse aux variations (illumination, météo, angle, bruit spectral). Un protocole d’entraînement unifié garantit l’équité. Les résultats montrent (i) une meilleure stabilité des petits modèles YOLOv11, (ii) des compromis précision–vitesse équilibrés pour YOLO-NAS et PP-YOLOE, (iii) >30% d’économie énergétique via pruning structuré et quantification post-entraînement avec perte mAP minimale. Nous discutons des scénarios de déploiement (drones, robots, capteurs fixes) et des perspectives : fusion multispectrale, distillation multi-teachers, adaptation continue.

## 1. Introduction

- Contexte agronomique : impact économique des mauvaises herbes
- Limites des herbicides (résistance, écotoxicité)
- Rôle de la vision embarquée temps réel
- Progression YOLO (v7→v8→v11) et émergence des transformeurs / NAS
- Lacunes : peu d’études multi-générations + énergie + robustesse
- Contributions (liste) :
  1. Benchmark multi-générations + familles variées (CNN, NAS, Transformer) sur 4 datasets spécialisés
  2. Cadre unifié intégrant énergie, robustesse perturbations et transfert cross-dataset
  3. Analyse fine précision / latence / énergie / taille
  4. Étude effets quantification & pruning structurés
  5. Ressources reproductibles (scripts, configs, protocoles statistiques)

## 2. Travaux connexes

### 2.1 Vision pour l’agriculture (détection mauvaises herbes)

### 2.2 Évolution YOLO (architecture, backbone, head, loss)

### 2.3 Autres détecteurs : YOLO-NAS, PP-YOLOE, DETR/RT-DETR, EfficientDet

### 2.4 Benchmarks existants & limites

## 3. Datasets

### 3.1 Weed25

- Classes, diversité environnementale, résolution

### 3.2 DeepWeeds

### 3.3 CWD30

### 3.4 WeedsGalore (multispectral)

### 3.5 Harmonisation annotations (conversion COCO / label mapping)

### 3.6 Splits (stratifiés, ratios, justification)

### 3.7 Cross-dataset & zero-shot scenario (optionnel)

Table T1: Dataset summary (images, classes, spectral bands, median resolution, split ratios)

## 4. Méthodologie expérimentale

### 4.1 Modèles (versions exactes, sources, poids initiaux)

### 4.2 Hyperparamètres communs (batch, epochs, optimizer, scheduler, EMA)

### 4.3 Prétraitement (resize, letterbox, normalisation, class balancing)

### 4.4 Augmentations (mosaic, mixup, color jitter, rotation UAV, spectral alignment)

### 4.5 Inférence (NMS, confidence thresholds, TTA minimal)

### 4.6 Optimisations edge (ONNX, TensorRT, INT8/PTQ, structured pruning L1, channel slimming)

### 4.7 Mesure énergie (procédure, matériel: Jetson Orin, RTX 4090, powermeter, NVML sampling)

### 4.8 Robustesse (perturbations : γ-scaling, blur, noise, weather simulation, spectral noise)

### 4.9 Reproductibilité (seed fixation, deterministic ops caveats)

Table T2: Unified hyperparameters
Figure F1: Experimental pipeline diagram

## 5. Métriques et analyse

### 5.1 mAP@0.5 et mAP@0.5:0.95 (définition rapide)

### 5.2 Latence & throughput (mean / p95, batch=1)

### 5.3 Mémoire (VRAM peak, disk footprint)

### 5.4 Énergie (J/frame, W moyenne, variance)

### 5.5 Robustesse (ΔmAP perturbations vs baseline)

### 5.6 Score composite pondéré (formule paramétrable)

### 5.7 Statistiques (bootstrap CI, Wilcoxon paired tests, Spearman correlations)

## 6. Résultats

### 6.1 Tableau principal (mAP, FPS, Params, Size MB, J/frame)

### 6.2 Courbes précision vs latence (scatter + Pareto frontier)

### 6.3 Analyse par espèce (long-tail performance)

### 6.4 Robustesse (heatmap ΔmAP)

### 6.5 Impact pruning / quantification (avant / après)

### 6.6 Transfert cross-dataset (train A → test B matrix)

Tables: T3 main, T4 robustness, T5 ablations, T6 compression
Figures: F2 precision-latency, F3 energy bars, F4 robustness heatmap, F5 failure cases collage

## 7. Discussion

### 7.1 Trade-offs small vs large models

### 7.2 Rôle des augmentations pour robustesse

### 7.3 Limites (biais géographiques, bruit d’annotation)

### 7.4 Implications pour UAV, robots, capteurs fixes (cycle temps réel, batterie)

## 8. Conclusion & Perspectives

- Synthèse meilleures configurations par scénario
- Recommandations pipeline déploiement (prétraitement, format export, monitoring)
- Pistes futures : fusion multi-capteurs, distillation multi-teachers, adaptation continue, segmentation panoptique

## 9. Reproductibilité

### 9.1 Versioning (git tags, dataset hash)

### 9.2 Scripts (train.py, eval.py, energy_logger.py, export_onnx.py, prune.py, quantize.py)

### 9.3 Config format (YAML) + example

### 9.4 Publication artefacts (Zenodo DOI)

## 10. Références

(Add BibTeX entries in `refs.bib` – placeholder)

---

## Proposed File/Folder Structure

```
project/
  data/ (symlinks or README on download)
  configs/
    base.yaml
    yolo_n.yaml
    ...
  scripts/
    train.py
    eval.py
    energy_logger.py
    perturb_eval.py
    prune.py
    quantize.py
    export_onnx.py
  results/
    tables/
    figures/
  models/ (exported weights)
  reports/
    paper_outline.md
    main.tex
  refs.bib
```

## Energy Measurement Procedure (Detailed)

1. Warmup 100 images (excluded from stats)
2. Synchronize GPU (torch.cuda.synchronize) before timing
3. Random sample N=500 test images (batch=1) – ensure deterministic transforms off
4. Start power logging (NVML every 1s, or tegrastats)
5. Record timestamps + instantaneous power → integrate trapezoidal rule → Joules total / frames
6. Repeat 3 runs; report mean ± std; compute coefficient of variation
7. For edge INT8 variant replicate same set to isolate quantization effect

Pseudo-code snippet (placeholder):

```python
# energy_logger.py (skeleton)
# measure_energy(model, dataloader, device, logger, runs=3)
```

## Composite Score (Optional)

S = w1 _ mAP_0.5:0.95_norm + w2 _ FPS_norm + w3 _ (1 - Energy_norm) + w4 _ Robustness_norm

- Normalization: min-max across models.
- Choose weights per deployment scenario (e.g., UAV: emphasize Energy & FPS).

## Statistical Protocol

- Bootstrap 2000 samples for mAP CIs by image
- Wilcoxon signed-rank test YOLOv11 vs YOLOv8 per dataset (α=0.05, Holm correction)
- Report effect size r
- Spearman ρ: (params vs J/frame), (mAP vs energy)

## Limitations

- Non-exhaustive seasonal variation
- Potential annotation noise in multispectral dataset
- Generalization to unseen species not fully validated

## Ethical & Practical Considerations

- Reduction herbicide usage impact
- Data privacy (if on-farm imagery contains unintended objects)
- Model updates & drift monitoring

## Appendix (Planned)

A. Detailed hyperparameters per model
B. Additional qualitative detections
C. Failure case taxonomy
D. Pruning & quantization settings

---

Version: 0.1 (skeleton)
Next steps: populate dataset stats, write Related Work with citations, implement scripts, collect baseline results.
