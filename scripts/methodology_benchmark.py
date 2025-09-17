#!/usr/bin/env python3
"""
Script de benchmark complet pour la méthodologie de l'article.
Lance l'entraînement et l'évaluation de tous les modèles sur tous les datasets 
avec un protocole unifié.

Modèles testés:
- YOLOv8 (n, s, m, l, x)
- YOLOv11 (n, s, m, l, x) 
- YOLO-NAS (s, m, l)
- YOLOX (nano, tiny, s, m, l, x)
- YOLOv7 (tiny, s, m, l, x)
- PP-YOLOE (s, m, l, x)
- EfficientDet (d0-d7)
- DETR (base, large)
- RT-DETR (l, x)

Datasets:
- Weed25 (25 espèces de mauvaises herbes)
- DeepWeeds (8 espèces, environnements variés)
- CWD30 (20 espèces + cultures)
- WeedsGalore (UAV multispectral, segmentation)

Métriques collectées:
- mAP@0.5, mAP@0.5:0.95
- FPS, latence (P95)
- Taille du modèle (paramètres, MB)
- Consommation énergétique
"""

import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class MethodologyBenchmark:
    """Gestionnaire de benchmark pour la méthodologie de l'article."""
    
    def __init__(self, device: str = 'auto', epochs: int = 100, batch_size: int = 16):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_dir = Path('results/methodology_benchmark')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des modèles par famille
        self.model_configs = {
            'yolov8': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
            'yolov11': ['yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt'],
            'yolo_nas': ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
            'yolox': ['yolox_nano', 'yolox_tiny', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'],
            'yolov7': ['yolov7.pt'],
            'pp_yoloe': ['ppyoloe_s', 'ppyoloe_m', 'ppyoloe_l', 'ppyoloe_x'],
            'efficientdet': ['efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 
                           'efficientdet_d3', 'efficientdet_d4'],
            'detr': ['detr_resnet50', 'detr_resnet101'],
            'rt_detr': ['rtdetr_l', 'rtdetr_x']
        }
        
        # Configuration des datasets
        self.datasets = {
            'weed25': 'data/weed25.yaml',
            'deepweeds': 'data/deepweeds.yaml', 
            'cwd30': 'data/cwd30.yaml',
            'weedsgalore': 'data/weedsgalore.yaml'
        }
        
        # Vérifier les datasets disponibles
        self.available_datasets = self._check_datasets()
        
    def _check_datasets(self) -> Dict[str, str]:
        """Vérifie quels datasets sont disponibles."""
        available = {}
        for name, yaml_path in self.datasets.items():
            if Path(yaml_path).exists():
                # Vérifier si les données existent aussi
                data_dir = Path(f'data/{name}')
                if data_dir.exists() and any(data_dir.iterdir()):
                    available[name] = yaml_path
                else:
                    print(f"⚠️  Dataset {name}: YAML trouvé mais données manquantes")
            else:
                print(f"⚠️  Dataset {name}: Configuration YAML manquante")
        
        if not available:
            print("⚠️  Aucun dataset disponible, utilisation du dataset dummy")
            available['dummy'] = 'data/dummy.yaml'
            
        return available
    
    def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles disponibles selon les dépendances."""
        available_models = []
        
        # YOLOv8/v11 toujours disponibles (ultralytics)
        available_models.extend(self.model_configs['yolov8'])
        available_models.extend(self.model_configs['yolov11'])
        
        # Vérifier les autres dépendances
        try:
            import super_gradients
            available_models.extend(self.model_configs['yolo_nas'])
        except ImportError:
            print("⚠️  YOLO-NAS non disponible (super-gradients manquant)")
            
        try:
            import yolox
            available_models.extend(self.model_configs['yolox'])
        except ImportError:
            print("⚠️  YOLOX non disponible (yolox manquant)")
            
        # PP-YOLOE, EfficientDet, DETR nécessitent des packages spéciaux
        # On les garde dans la liste mais ils échoueront gracieusement
        
        return available_models
    
    def train_model(self, model_name: str, dataset_name: str, dataset_yaml: str) -> Dict[str, Any]:
        """Entraîne un modèle sur un dataset."""
        print(f"\n🚀 Entraînement: {model_name} sur {dataset_name}")
        
        # Créer le répertoire de sortie
        output_dir = self.results_dir / f"{model_name}_{dataset_name}"
        
        # Commande d'entraînement
        cmd = [
            sys.executable, 'scripts/train.py',
            '--models', model_name,
            '--data', dataset_yaml,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--device', self.device,
            '--output', str(output_dir)
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1h timeout
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Entraînement réussi en {training_time:.1f}s")
                
                # Trouver le fichier de poids (chercher dans output_dir)
                # Le script train.py sauvegarde les modèles dans output_dir/model_name/
                model_dirs = list(output_dir.glob('**/'))
                best_weights = None
                
                for model_dir in model_dirs:
                    weights_dir = model_dir / 'weights'
                    if weights_dir.exists():
                        best_path = weights_dir / 'best.pt'
                        if best_path.exists():
                            best_weights = best_path
                            break
                
                return {
                    'status': 'success',
                    'training_time': training_time,
                    'weights_path': str(best_weights) if best_weights else None,
                    'output_dir': str(output_dir)
                }
            else:
                print(f"❌ Échec de l'entraînement: {result.stderr}")
                return {
                    'status': 'failed',
                    'error': result.stderr,
                    'training_time': training_time
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout de l'entraînement après 1h")
            return {
                'status': 'timeout',
                'training_time': 3600
            }
        except Exception as e:
            print(f"💥 Erreur lors de l'entraînement: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'training_time': 0
            }
    
    def evaluate_model(self, weights_path: str, dataset_yaml: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Évalue un modèle entraîné."""
        print(f"📊 Évaluation: {model_name} sur {dataset_name}")
        
        cmd = [
            sys.executable, 'scripts/evaluate.py',
            '--weights', weights_path,
            '--data', dataset_yaml,
            '--device', self.device,
            '--output', str(self.results_dir / f"eval_{model_name}_{dataset_name}.json")
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Évaluation réussie")
                
                # Charger les résultats
                eval_file = self.results_dir / f"eval_{model_name}_{dataset_name}.json"
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        eval_results = json.load(f)
                    return eval_results[0] if eval_results else {}
                else:
                    return {'status': 'no_results'}
            else:
                print(f"❌ Échec de l'évaluation: {result.stderr}")
                return {'status': 'eval_failed', 'error': result.stderr}
                
        except Exception as e:
            print(f"💥 Erreur lors de l'évaluation: {e}")
            return {'status': 'eval_error', 'error': str(e)}
    
    def run_full_benchmark(self, models: List[str] = None, datasets: List[str] = None) -> Dict[str, Any]:
        """Lance le benchmark complet."""
        
        # Modèles à tester
        if models is None:
            models = self.get_available_models()
        
        # Datasets à tester
        if datasets is None:
            datasets = list(self.available_datasets.keys())
        
        print(f"\n🎯 BENCHMARK MÉTHODOLOGIE")
        print(f"📦 Modèles: {len(models)} ({', '.join(models[:5])}...)")
        print(f"📊 Datasets: {len(datasets)} ({', '.join(datasets)})")
        print(f"⚙️  Paramètres: epochs={self.epochs}, batch_size={self.batch_size}")
        print(f"🖥️  Device: {self.device}")
        
        # Résultats globaux
        all_results = []
        total_experiments = len(models) * len(datasets)
        current_experiment = 0
        
        for dataset_name in datasets:
            dataset_yaml = self.available_datasets[dataset_name]
            
            for model_name in models:
                current_experiment += 1
                print(f"\n{'='*60}")
                print(f"Expérience {current_experiment}/{total_experiments}: {model_name} × {dataset_name}")
                print(f"{'='*60}")
                
                # Entraînement
                train_result = self.train_model(model_name, dataset_name, dataset_yaml)
                
                # Résultat de base
                experiment_result = {
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'dataset_yaml': dataset_yaml,
                    'training_status': train_result['status'],
                    'training_time': train_result.get('training_time', 0),
                    'experiment_id': f"{model_name}_{dataset_name}",
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Évaluation si l'entraînement a réussi
                if train_result['status'] == 'success' and train_result.get('weights_path'):
                    eval_result = self.evaluate_model(
                        train_result['weights_path'], 
                        dataset_yaml, 
                        model_name, 
                        dataset_name
                    )
                    experiment_result.update(eval_result)
                else:
                    experiment_result.update({
                        'map50': 0.0,
                        'map50_95': 0.0,
                        'fps': 0.0,
                        'total_parameters': 0,
                        'model_size_mb': 0.0
                    })
                
                all_results.append(experiment_result)
                
                # Sauvegarde intermédiaire
                self._save_results(all_results)
        
        print(f"\n🎉 BENCHMARK TERMINÉ!")
        print(f"📈 {len(all_results)} expériences réalisées")
        
        # Générer le rapport final
        self._generate_report(all_results)
        
        return {
            'total_experiments': len(all_results),
            'results': all_results,
            'summary_file': str(self.results_dir / 'methodology_results.json'),
            'report_file': str(self.results_dir / 'methodology_report.csv')
        }
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Sauvegarde les résultats."""
        output_file = self.results_dir / 'methodology_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Génère un rapport CSV pour l'analyse."""
        df = pd.DataFrame(results)
        
        # Colonnes principales pour l'article
        columns_order = [
            'model_name', 'dataset_name', 'training_status',
            'map50', 'map50_95', 'fps', 'total_parameters', 
            'model_size_mb', 'training_time'
        ]
        
        # Réorganiser les colonnes
        existing_cols = [col for col in columns_order if col in df.columns]
        df_report = df[existing_cols]
        
        # Sauvegarder
        report_file = self.results_dir / 'methodology_report.csv'
        df_report.to_csv(report_file, index=False)
        
        print(f"📊 Rapport sauvegardé: {report_file}")
        
        # Statistiques rapides
        successful = df[df['training_status'] == 'success']
        print(f"✅ Entraînements réussis: {len(successful)}/{len(df)}")
        
        if len(successful) > 0:
            print(f"📈 mAP@0.5 moyen: {successful['map50'].mean():.3f}")
            print(f"⚡ FPS moyen: {successful['fps'].mean():.1f}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark complet pour méthodologie')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Nombre d\'époques pour l\'entraînement (défaut: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Taille du batch (défaut: 16)')
    parser.add_argument('--device', default='auto',
                       help='Device pour l\'entraînement (auto/cpu/cuda)')
    parser.add_argument('--models', nargs='*',
                       help='Modèles spécifiques à tester (défaut: tous)')
    parser.add_argument('--datasets', nargs='*', 
                       help='Datasets spécifiques à tester (défaut: tous)')
    parser.add_argument('--quick', action='store_true',
                       help='Test rapide (5 époques, YOLOv8n uniquement)')
    
    args = parser.parse_args()
    
    # Mode test rapide
    if args.quick:
        args.epochs = 5
        args.models = ['yolov8n.pt']
        print("🚀 Mode test rapide activé (5 époques, YOLOv8n seulement)")
    
    # Créer le benchmark
    benchmark = MethodologyBenchmark(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Lancer le benchmark
    results = benchmark.run_full_benchmark(
        models=args.models,
        datasets=args.datasets
    )
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"📁 Résultats: {results['summary_file']}")
    print(f"📊 Rapport: {results['report_file']}")

if __name__ == '__main__':
    main()