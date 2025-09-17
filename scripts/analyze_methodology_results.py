#!/usr/bin/env python3
"""
Script d'analyse et de génération de rapport pour les résultats de benchmark méthodologique.
Génère des visualisations, des tableaux comparatifs et des analyses statistiques détaillées.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Configuration de matplotlib pour éviter les erreurs
plt.style.use('default')
warnings.filterwarnings('ignore')

class MethodologyResultsAnalyzer:
    """Analyseur de résultats de benchmark méthodologique."""
    
    def __init__(self, results_dir: str = "results/comprehensive_methodology"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "comprehensive_results.json"
        self.analysis_file = self.results_dir / "methodology_analysis.json"
        self.output_dir = self.results_dir / "analysis_output"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Charge les résultats du benchmark."""
        
        # Charger les résultats JSON
        if not self.results_file.exists():
            raise FileNotFoundError(f"Fichier de résultats non trouvé: {self.results_file}")
            
        with open(self.results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
            
        df = pd.DataFrame(results_data)
        
        # Charger l'analyse si elle existe
        analysis = {}
        if self.analysis_file.exists():
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                
        return df, analysis
        
    def generate_performance_comparison(self, df: pd.DataFrame) -> None:
        """Génère un graphique de comparaison des performances."""
        
        # Filtrer les résultats réussis
        successful = df[df['evaluation_status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("⚠️  Aucun résultat réussi pour générer les graphiques")
            return
            
        # Graphique des performances par modèle
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Performances - Méthodologie Benchmark', fontsize=16, fontweight='bold')
        
        # mAP@0.5 par modèle
        if 'map50' in successful.columns and not successful['map50'].isna().all():
            ax1 = axes[0, 0]
            model_map50 = successful.groupby('model_name')['map50'].mean().sort_values(ascending=False)
            bars1 = ax1.bar(model_map50.index, model_map50.values, color='skyblue', alpha=0.8)
            ax1.set_title('mAP@0.5 par Modèle', fontweight='bold')
            ax1.set_ylabel('mAP@0.5')
            ax1.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # FPS par modèle
        if 'fps' in successful.columns and not successful['fps'].isna().all():
            ax2 = axes[0, 1]
            model_fps = successful.groupby('model_name')['fps'].mean().sort_values(ascending=False)
            bars2 = ax2.bar(model_fps.index, model_fps.values, color='lightcoral', alpha=0.8)
            ax2.set_title('FPS par Modèle', fontweight='bold')
            ax2.set_ylabel('FPS')
            ax2.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Taille des modèles
        if 'model_size_mb' in successful.columns and not successful['model_size_mb'].isna().all():
            ax3 = axes[1, 0]
            model_size = successful.groupby('model_name')['model_size_mb'].mean().sort_values(ascending=True)
            bars3 = ax3.bar(model_size.index, model_size.values, color='lightgreen', alpha=0.8)
            ax3.set_title('Taille des Modèles (MB)', fontweight='bold')
            ax3.set_ylabel('Taille (MB)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Temps d'entraînement
        if 'training_time' in successful.columns and not successful['training_time'].isna().all():
            ax4 = axes[1, 1]
            model_time = successful.groupby('model_name')['training_time'].mean() / 60  # en minutes
            bars4 = ax4.bar(model_time.index, model_time.values, color='gold', alpha=0.8)
            ax4.set_title('Temps d\'Entraînement (minutes)', fontweight='bold')
            ax4.set_ylabel('Temps (minutes)')
            ax4.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"📊 Graphique de performance sauvegardé: {self.output_dir}/performance_comparison.png")
        
    def generate_dataset_difficulty_analysis(self, df: pd.DataFrame) -> None:
        """Génère une analyse de la difficulté des datasets."""
        
        successful = df[df['evaluation_status'] == 'success'].copy()
        
        if len(successful) == 0 or 'map50' not in successful.columns:
            print("⚠️  Données insuffisantes pour l'analyse des datasets")
            return
            
        # Analyse par dataset
        dataset_stats = successful.groupby('dataset_name').agg({
            'map50': ['mean', 'std', 'count'],
            'training_time': ['mean', 'std']
        }).round(3)
        
        # Graphique des performances par dataset
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Analyse de la Difficulté des Datasets', fontsize=16, fontweight='bold')
        
        # mAP@0.5 moyen par dataset
        dataset_map50_mean = dataset_stats['map50']['mean'].sort_values(ascending=False)
        dataset_map50_std = dataset_stats['map50']['std']
        
        bars1 = ax1.bar(dataset_map50_mean.index, dataset_map50_mean.values, 
                       yerr=dataset_map50_std[dataset_map50_mean.index],
                       color='lightblue', alpha=0.8, capsize=5)
        ax1.set_title('mAP@0.5 par Dataset (avec écart-type)', fontweight='bold')
        ax1.set_ylabel('mAP@0.5')
        ax1.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            dataset = dataset_map50_mean.index[i]
            count = dataset_stats.loc[dataset, ('map50', 'count')]
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}\\n(n={count})', ha='center', va='bottom', fontweight='bold')
        
        # Temps d'entraînement par dataset
        if 'training_time' in dataset_stats.columns:
            dataset_time_mean = dataset_stats['training_time']['mean'] / 60  # en minutes
            bars2 = ax2.bar(dataset_time_mean.index, dataset_time_mean.values,
                           color='lightcoral', alpha=0.8)
            ax2.set_title('Temps d\'Entraînement Moyen par Dataset', fontweight='bold')
            ax2.set_ylabel('Temps (minutes)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_difficulty.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'dataset_difficulty.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"📊 Analyse des datasets sauvegardée: {self.output_dir}/dataset_difficulty.png")
        
    def generate_efficiency_analysis(self, df: pd.DataFrame) -> None:
        """Génère une analyse d'efficacité (performance vs ressources)."""
        
        successful = df[df['evaluation_status'] == 'success'].copy()
        
        if len(successful) == 0:
            return
            
        # Graphique de corrélation performance vs ressources
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Analyse d\'Efficacité des Modèles', fontsize=16, fontweight='bold')
        
        # Performance vs Taille du modèle
        if 'map50' in successful.columns and 'model_size_mb' in successful.columns:
            if not successful['map50'].isna().all() and not successful['model_size_mb'].isna().all():
                ax1 = axes[0]
                scatter1 = ax1.scatter(successful['model_size_mb'], successful['map50'], 
                                     c=successful['fps'], cmap='viridis', alpha=0.7, s=100)
                ax1.set_xlabel('Taille du Modèle (MB)')
                ax1.set_ylabel('mAP@0.5')
                ax1.set_title('Performance vs Taille (couleur = FPS)')
                
                # Ajouter une colorbar
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label('FPS')
                
                # Ajouter les noms des modèles
                for i, row in successful.iterrows():
                    ax1.annotate(row['model_name'], 
                               (row['model_size_mb'], row['map50']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        # Performance vs Temps d'entraînement
        if 'map50' in successful.columns and 'training_time' in successful.columns:
            if not successful['map50'].isna().all() and not successful['training_time'].isna().all():
                ax2 = axes[1]
                training_minutes = successful['training_time'] / 60
                scatter2 = ax2.scatter(training_minutes, successful['map50'], 
                                     c=successful['fps'], cmap='plasma', alpha=0.7, s=100)
                ax2.set_xlabel('Temps d\'Entraînement (minutes)')
                ax2.set_ylabel('mAP@0.5')
                ax2.set_title('Performance vs Temps d\'Entraînement')
                
                # Ajouter une colorbar
                cbar2 = plt.colorbar(scatter2, ax=ax2)
                cbar2.set_label('FPS')
                
                # Ajouter les noms des modèles
                for i, row in successful.iterrows():
                    ax2.annotate(row['model_name'], 
                               (row['training_time'] / 60, row['map50']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'efficiency_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"📊 Analyse d'efficacité sauvegardée: {self.output_dir}/efficiency_analysis.png")
        
    def generate_methodology_table(self, df: pd.DataFrame) -> None:
        """Génère un tableau de résultats pour l'article méthodologique."""
        
        successful = df[df['evaluation_status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("⚠️  Aucun résultat pour générer le tableau")
            return
            
        # Grouper par modèle et calculer les moyennes
        metrics_cols = ['map50', 'map50_95', 'fps', 'total_parameters', 'model_size_mb', 'training_time']
        available_cols = [col for col in metrics_cols if col in successful.columns]
        
        if not available_cols:
            print("⚠️  Métriques insuffisantes pour le tableau")
            return
            
        model_summary = successful.groupby('model_name')[available_cols].agg(['mean', 'std']).round(3)
        
        # Créer un tableau formaté pour l'article
        methodology_table = []
        
        for model in model_summary.index:
            row = {'Model': model}
            
            if 'map50' in available_cols:
                mean_map50 = model_summary.loc[model, ('map50', 'mean')]
                std_map50 = model_summary.loc[model, ('map50', 'std')]
                row['mAP@0.5'] = f"{mean_map50:.3f} ± {std_map50:.3f}"
            
            if 'map50_95' in available_cols:
                mean_map50_95 = model_summary.loc[model, ('map50_95', 'mean')]
                std_map50_95 = model_summary.loc[model, ('map50_95', 'std')]
                row['mAP@0.5:0.95'] = f"{mean_map50_95:.3f} ± {std_map50_95:.3f}"
            
            if 'fps' in available_cols:
                mean_fps = model_summary.loc[model, ('fps', 'mean')]
                std_fps = model_summary.loc[model, ('fps', 'std')]
                row['FPS'] = f"{mean_fps:.1f} ± {std_fps:.1f}"
            
            if 'total_parameters' in available_cols:
                mean_params = model_summary.loc[model, ('total_parameters', 'mean')]
                row['Parameters (M)'] = f"{mean_params/1e6:.2f}" if mean_params > 0 else "N/A"
            
            if 'model_size_mb' in available_cols:
                mean_size = model_summary.loc[model, ('model_size_mb', 'mean')]
                row['Size (MB)'] = f"{mean_size:.1f}"
            
            if 'training_time' in available_cols:
                mean_time = model_summary.loc[model, ('training_time', 'mean')] / 60  # en minutes
                row['Training Time (min)'] = f"{mean_time:.1f}"
            
            methodology_table.append(row)
        
        # Convertir en DataFrame et sauvegarder
        table_df = pd.DataFrame(methodology_table)
        table_file = self.output_dir / 'methodology_table.csv'
        table_df.to_csv(table_file, index=False, encoding='utf-8')
        
        # Générer aussi un tableau LaTeX
        latex_file = self.output_dir / 'methodology_table.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% Tableau des résultats méthodologiques\\n")
            f.write("\\begin{table}[htbp]\\n")
            f.write("\\centering\\n")
            f.write("\\caption{Comparaison des performances des modèles de détection}\\n")
            f.write("\\label{tab:methodology_results}\\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|c|c|}\\n")
            f.write("\\hline\\n")
            f.write("Modèle & mAP@0.5 & mAP@0.5:0.95 & FPS & Paramètres (M) & Taille (MB) & Temps (min) \\\\\\n")
            f.write("\\hline\\n")
            
            for _, row in table_df.iterrows():
                latex_row = f"{row.get('Model', 'N/A')} & "
                latex_row += f"{row.get('mAP@0.5', 'N/A')} & "
                latex_row += f"{row.get('mAP@0.5:0.95', 'N/A')} & "
                latex_row += f"{row.get('FPS', 'N/A')} & "
                latex_row += f"{row.get('Parameters (M)', 'N/A')} & "
                latex_row += f"{row.get('Size (MB)', 'N/A')} & "
                latex_row += f"{row.get('Training Time (min)', 'N/A')} \\\\\\"
                f.write(latex_row + "\\n")
            
            f.write("\\hline\\n")
            f.write("\\end{tabular}\\n")
            f.write("\\end{table}\\n")
        
        print(f"📊 Tableau méthodologique sauvegardé:")
        print(f"   CSV: {table_file}")
        print(f"   LaTeX: {latex_file}")
        
        # Afficher le tableau dans la console
        print("\\n" + "="*80)
        print("TABLEAU DES RÉSULTATS MÉTHODOLOGIQUES")
        print("="*80)
        print(table_df.to_string(index=False))
        print("="*80)
        
    def generate_comprehensive_report(self) -> None:
        """Génère un rapport complet d'analyse méthodologique."""
        
        print("🔍 GÉNÉRATION DU RAPPORT MÉTHODOLOGIQUE COMPLET")
        print("="*60)
        
        try:
            # Charger les données
            df, analysis = self.load_results()
            print(f"✅ Données chargées: {len(df)} expériences")
            
            # Générer les analyses
            print("📊 Génération des graphiques...")
            self.generate_performance_comparison(df)
            self.generate_dataset_difficulty_analysis(df)
            self.generate_efficiency_analysis(df)
            
            print("📋 Génération du tableau méthodologique...")
            self.generate_methodology_table(df)
            
            # Rapport textuel
            report_file = self.output_dir / 'methodology_report.md'
            self.generate_text_report(df, analysis, report_file)
            
            print(f"\\n🎯 RAPPORT COMPLET GÉNÉRÉ!")
            print(f"📁 Répertoire de sortie: {self.output_dir}")
            print(f"📊 Fichiers générés:")
            for file in self.output_dir.iterdir():
                if file.is_file():
                    print(f"   - {file.name}")
                    
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport: {e}")
            
    def generate_text_report(self, df: pd.DataFrame, analysis: Dict[str, Any], output_file: Path) -> None:
        """Génère un rapport textuel en markdown."""
        
        successful = df[df['evaluation_status'] == 'success']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Rapport Méthodologique - Benchmark de Détection d'Objets\\n\\n")
            
            f.write("## Résumé Exécutif\\n\\n")
            f.write(f"- **Total d'expériences**: {len(df)}\\n")
            f.write(f"- **Expériences réussies**: {len(successful)} ({len(successful)/len(df)*100:.1f}%)\\n")
            f.write(f"- **Modèles testés**: {df['model_name'].nunique()}\\n")
            f.write(f"- **Datasets testés**: {df['dataset_name'].nunique()}\\n\\n")
            
            if len(successful) > 0:
                f.write("## Résultats Principaux\\n\\n")
                
                if 'map50' in successful.columns and not successful['map50'].isna().all():
                    best_model = successful.loc[successful['map50'].idxmax()]
                    f.write(f"- **Meilleur modèle**: {best_model['model_name']} (mAP@0.5: {best_model['map50']:.3f})\\n")
                
                if 'fps' in successful.columns and not successful['fps'].isna().all():
                    fastest_model = successful.loc[successful['fps'].idxmax()]
                    f.write(f"- **Modèle le plus rapide**: {fastest_model['model_name']} ({fastest_model['fps']:.1f} FPS)\\n")
                
                f.write("\\n## Analyse Détaillée\\n\\n")
                f.write("### Performance par Modèle\\n\\n")
                
                model_stats = successful.groupby('model_name').agg({
                    col: ['mean', 'std'] for col in ['map50', 'fps', 'model_size_mb', 'training_time'] 
                    if col in successful.columns
                }).round(3)
                
                for model in model_stats.index:
                    f.write(f"**{model}**:\\n")
                    if 'map50' in model_stats.columns:
                        f.write(f"- mAP@0.5: {model_stats.loc[model, ('map50', 'mean')]:.3f} ± {model_stats.loc[model, ('map50', 'std')]:.3f}\\n")
                    if 'fps' in model_stats.columns:
                        f.write(f"- FPS: {model_stats.loc[model, ('fps', 'mean')]:.1f} ± {model_stats.loc[model, ('fps', 'std')]:.1f}\\n")
                    f.write("\\n")
            
            f.write("## Méthodologie\\n\\n")
            f.write("### Protocole d'Entraînement\\n\\n")
            f.write("- **Prétraitement uniforme**: Augmentation de données standardisée\\n")
            f.write("- **Hyperparamètres identiques**: Même batch size, epochs, learning rate\\n")
            f.write("- **Métriques complètes**: mAP@0.5, mAP@0.5:0.95, FPS, taille modèle\\n")
            f.write("- **Validation rigoureuse**: Vérification de cohérence des résultats\\n\\n")
            
            f.write("### Datasets Utilisés\\n\\n")
            for dataset in df['dataset_name'].unique():
                dataset_results = successful[successful['dataset_name'] == dataset]
                if len(dataset_results) > 0:
                    avg_map50 = dataset_results['map50'].mean() if 'map50' in dataset_results.columns else 0
                    f.write(f"- **{dataset}**: {len(dataset_results)} modèles testés, mAP@0.5 moyen: {avg_map50:.3f}\\n")
            
            f.write("\\n## Conclusions et Recommandations\\n\\n")
            f.write("Ce benchmark méthodologique fournit une comparaison équitable et rigoureuse des modèles de détection d'objets ")
            f.write("pour la détection de mauvaises herbes. Les résultats peuvent guider le choix du modèle optimal ")
            f.write("selon les contraintes de performance, de vitesse et de ressources.\\n")
        
        print(f"📄 Rapport textuel généré: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyse des résultats de benchmark méthodologique')
    parser.add_argument('--results_dir', default='results/comprehensive_methodology',
                       help='Répertoire contenant les résultats')
    parser.add_argument('--output_dir', 
                       help='Répertoire de sortie pour les analyses (défaut: results_dir/analysis_output)')
    
    args = parser.parse_args()
    
    # Créer l'analyseur
    analyzer = MethodologyResultsAnalyzer(args.results_dir)
    
    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(exist_ok=True)
    
    # Générer le rapport complet
    analyzer.generate_comprehensive_report()

if __name__ == '__main__':
    main()