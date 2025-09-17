#!/usr/bin/env python3
"""
Générateur de visualisations pour la section Résultats et Discussion
==================================================================

Ce script génère les graphiques et tableaux pour l'analyse comparative
des performances des modèles de détection de mauvaises herbes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_performance_comparison_charts():
    """Crée les graphiques de comparaison de performances."""
    
    # Données des résultats réels
    data = {
        'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv11n', 'YOLOv8n', 'YOLOv8s', 'YOLOv8n', 'YOLOv8n'],
        'Dataset': ['DeepWeeds', 'DeepWeeds', 'DeepWeeds', 'Weed25', 'Weed25', 'CWD30', 'WeedsGalore'],
        'mAP50': [0.366, 1.25, 0.89, 0.24, 0.87, 0.18, 1.56],
        'FPS': [6.31, 4.08, 7.03, 6.03, 3.98, 5.81, 6.73],
        'ModelSize_MB': [5.95, 21.5, 5.2, 6.12, 21.8, 6.25, 5.89],
        'Memory_MB': [22.94, 45.7, 21.8, 24.1, 47.2, 25.3, 22.1],
        'InferenceTime_ms': [158.5, 245.2, 142.3, 165.8, 251.4, 172.1, 148.7]
    }
    
    df = pd.DataFrame(data)
    
    # Créer le répertoire de sortie
    output_dir = Path("results/discussion_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Graphique mAP@50 vs FPS (Trade-off précision/vitesse)
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: mAP vs FPS
    plt.subplot(2, 2, 1)
    colors = {'YOLOv8n': '#FF6B6B', 'YOLOv8s': '#4ECDC4', 'YOLOv11n': '#45B7D1'}
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.scatter(model_data['FPS'], model_data['mAP50'], 
                   label=model, color=colors.get(model, '#95A5A6'), 
                   s=100, alpha=0.7, edgecolors='black')
        
        # Annotations des datasets
        for _, row in model_data.iterrows():
            plt.annotate(row['Dataset'], 
                        (row['FPS'], row['mAP50']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.xlabel('FPS (Images par seconde)')
    plt.ylabel('mAP@50 (%)')
    plt.title('Trade-off Précision vs Vitesse')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Efficacité mémoire vs Performance
    plt.subplot(2, 2, 2)
    df['Efficiency'] = df['mAP50'] / df['Memory_MB'] * 100  # Efficacité mémoire
    
    bars = plt.bar(range(len(df)), df['Efficiency'], 
                   color=[colors.get(model, '#95A5A6') for model in df['Model']])
    plt.xlabel('Configuration Modèle-Dataset')
    plt.ylabel('Efficacité Mémoire (mAP/MB)')
    plt.title('Efficacité Mémoire par Configuration')
    plt.xticks(range(len(df)), 
               [f"{row['Model'][:7]}\n{row['Dataset'][:8]}" for _, row in df.iterrows()],
               rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Taille modèle vs Performance
    plt.subplot(2, 2, 3)
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.scatter(model_data['ModelSize_MB'], model_data['mAP50'], 
                   label=model, color=colors.get(model, '#95A5A6'), 
                   s=100, alpha=0.7, edgecolors='black')
    
    plt.xlabel('Taille Modèle (MB)')
    plt.ylabel('mAP@50 (%)')
    plt.title('Performance vs Taille Modèle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Temps d'inférence par dataset
    plt.subplot(2, 2, 4)
    dataset_times = df.groupby('Dataset')['InferenceTime_ms'].mean()
    bars = plt.bar(dataset_times.index, dataset_times.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.xlabel('Dataset')
    plt.ylabel('Temps d\'inférence moyen (ms)')
    plt.title('Latence par Dataset')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    # 2. Graphique de complexité des datasets
    plt.figure(figsize=(10, 6))
    
    dataset_complexity = {
        'WeedsGalore': {'classes': 10, 'train_images': 24, 'difficulty': 2, 'mAP_avg': 1.56},
        'DeepWeeds': {'classes': 8, 'train_images': 80, 'difficulty': 3, 'mAP_avg': 0.84},
        'Weed25': {'classes': 25, 'train_images': 80, 'difficulty': 4, 'mAP_avg': 0.55},
        'CWD30': {'classes': 30, 'train_images': 40, 'difficulty': 5, 'mAP_avg': 0.18}
    }
    
    datasets = list(dataset_complexity.keys())
    classes = [dataset_complexity[d]['classes'] for d in datasets]
    performance = [dataset_complexity[d]['mAP_avg'] for d in datasets]
    difficulty = [dataset_complexity[d]['difficulty'] for d in datasets]
    
    # Créer un scatter plot avec taille proportionnelle à la difficulté
    scatter = plt.scatter(classes, performance, 
                         s=[d*100 for d in difficulty], 
                         c=difficulty, cmap='RdYlBu_r', 
                         alpha=0.7, edgecolors='black')
    
    # Annotations
    for i, dataset in enumerate(datasets):
        plt.annotate(dataset, (classes[i], performance[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.xlabel('Nombre de Classes')
    plt.ylabel('Performance Moyenne mAP@50 (%)')
    plt.title('Complexité des Datasets vs Performance\n(Taille des bulles = Difficulté)')
    
    # Colorbar pour la difficulté
    cbar = plt.colorbar(scatter)
    cbar.set_label('Niveau de Difficulté')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'dataset_complexity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'dataset_complexity.pdf', bbox_inches='tight')
    plt.show()
    
    # 3. Radar chart pour comparaison multi-critères
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Critères d'évaluation (normalisés 0-10)
    criteria = ['Précision', 'Vitesse', 'Efficacité\nMémoire', 'Taille\nModèle', 'Stabilité']
    
    # Scores normalisés pour chaque modèle (0-10)
    models_scores = {
        'YOLOv8n': [3, 7, 6, 8, 7],    # Précision faible, vitesse bonne, efficace
        'YOLOv8s': [8, 4, 4, 3, 7],    # Précision élevée, vitesse faible, gros modèle
        'YOLOv11n': [6, 9, 9, 9, 9]    # Équilibré, moderne, efficace
    }
    
    # Angles pour le radar
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model, scores) in enumerate(models_scores.items()):
        scores += scores[:1]  # Fermer le cercle
        ax.plot(angles, scores, 'o-', linewidth=2, label=model, color=colors_radar[i])
        ax.fill(angles, scores, alpha=0.25, color=colors_radar[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.set_ylim(0, 10)
    ax.set_title('Comparaison Multi-Critères des Modèles\n(Échelle 0-10)', size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'radar_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques sauvegardés dans : {output_dir}")
    print("📊 Fichiers générés :")
    print("  - performance_comparison.png/pdf")
    print("  - dataset_complexity.png/pdf") 
    print("  - radar_comparison.png/pdf")

def create_context_adaptation_chart():
    """Crée le graphique d'adaptation contextuelle."""
    
    output_dir = Path("results/discussion_figures")
    
    # Données d'adaptation contextuelle
    contexts = ['Drone', 'Robot Agricole', 'Capteurs Fixes']
    criteria = ['Précision\nRequise', 'Vitesse\nCritique', 'Contraintes\nÉnergie', 
               'Robustesse\nMétéo', 'Coût\nDéploiement', 'Maintenance']
    
    # Scores par contexte (0-5 échelle de criticité)
    context_scores = {
        'Drone': [3, 5, 5, 2, 4, 4],
        'Robot Agricole': [5, 4, 3, 5, 5, 3], 
        'Capteurs Fixes': [2, 1, 3, 5, 1, 1]
    }
    
    # Créer heatmap
    plt.figure(figsize=(12, 6))
    
    # Préparer les données pour la heatmap
    heatmap_data = []
    for context in contexts:
        heatmap_data.append(context_scores[context])
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=criteria, index=contexts)
    
    # Créer la heatmap
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlBu_r', center=2.5, 
                cbar_kws={'label': 'Niveau de Criticité (0-5)'}, 
                square=True, linewidths=0.5)
    
    plt.title('Matrice de Criticité par Contexte d\'Application', size=14, weight='bold', pad=20)
    plt.xlabel('Critères d\'Évaluation')
    plt.ylabel('Contextes d\'Application')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'context_adaptation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'context_adaptation.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphique d'adaptation contextuelle sauvegardé dans : {output_dir}")

if __name__ == "__main__":
    print("🎨 Génération des visualisations pour Résultats et Discussion")
    print("=" * 60)
    
    create_performance_comparison_charts()
    create_context_adaptation_chart()
    
    print("\n🎉 Toutes les visualisations ont été générées avec succès !")
    print("📁 Fichiers disponibles dans : results/discussion_figures/")