#!/usr/bin/env python3
"""
Script de lancement rapide pour le benchmark méthodologique complet.
Permet de lancer facilement des expériences avec différents profils de test.
"""

import argparse
import subprocess
import sys
from pathlib import Path

class MethodologyLauncher:
    """Lanceur simplifié pour les benchmarks méthodologiques."""
    
    def __init__(self):
        self.available_models = [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
            'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt',
            'yolov7.pt'
        ]
        
        self.available_datasets = ['weed25', 'deepweeds', 'cwd30', 'weedsgalore', 'dummy']
        
        self.profiles = {
            'quick': {
                'description': 'Test rapide (5 époques, YOLOv8n seulement)',
                'epochs': 5,
                'models': ['yolov8n.pt'],
                'datasets': ['dummy'],
                'batch_size': 16
            },
            'development': {
                'description': 'Développement (10 époques, modèles nano/small)',
                'epochs': 10,
                'models': ['yolov8n.pt', 'yolov8s.pt', 'yolov11n.pt'],
                'datasets': ['weed25'],
                'batch_size': 8
            },
            'validation': {
                'description': 'Validation (20 époques, sélection de modèles)',
                'epochs': 20,
                'models': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov11n.pt', 'yolov11s.pt'],
                'datasets': ['weed25', 'deepweeds'],
                'batch_size': 8
            },
            'publication': {
                'description': 'Publication (50 époques, tous modèles disponibles)',
                'epochs': 50,
                'models': None,  # Tous les modèles disponibles
                'datasets': None,  # Tous les datasets disponibles
                'batch_size': 8
            },
            'full': {
                'description': 'Benchmark complet (100 époques, tous modèles, tous datasets)',
                'epochs': 100,
                'models': None,
                'datasets': None,
                'batch_size': 4  # Batch size réduit pour les gros modèles
            }
        }
    
    def show_profiles(self):
        """Affiche les profils disponibles."""
        print("📋 PROFILS DE BENCHMARK DISPONIBLES:")
        print("=" * 60)
        
        for name, config in self.profiles.items():
            print(f"\n🔧 {name.upper()}")
            print(f"   {config['description']}")
            print(f"   Époques: {config['epochs']}")
            print(f"   Modèles: {len(config['models']) if config['models'] else 'tous'}")
            print(f"   Datasets: {len(config['datasets']) if config['datasets'] else 'tous'}")
            print(f"   Batch size: {config['batch_size']}")
    
    def estimate_duration(self, profile_name: str) -> str:
        """Estime la durée du benchmark."""
        config = self.profiles[profile_name]
        
        num_models = len(config['models']) if config['models'] else len(self.available_models)
        num_datasets = len(config['datasets']) if config['datasets'] else len(self.available_datasets)
        epochs = config['epochs']
        
        # Estimation grossière: ~2min/époque pour YOLOv8n, plus pour les autres
        base_time_per_epoch = 2  # minutes
        model_factor = 1.5  # Facteur moyen pour les modèles plus gros
        
        estimated_minutes = num_models * num_datasets * epochs * base_time_per_epoch * model_factor
        
        if estimated_minutes < 60:
            return f"~{estimated_minutes:.0f} minutes"
        elif estimated_minutes < 1440:  # 24h
            return f"~{estimated_minutes/60:.1f} heures"
        else:
            return f"~{estimated_minutes/1440:.1f} jours"
    
    def launch_benchmark(self, profile_name: str, custom_args: dict = None) -> bool:
        """Lance un benchmark avec le profil spécifié."""
        
        if profile_name not in self.profiles:
            print(f"❌ Profil '{profile_name}' non reconnu")
            return False
        
        config = self.profiles[profile_name]
        
        # Construire la commande
        cmd = [sys.executable, 'scripts/comprehensive_methodology_benchmark.py']
        
        # Ajouter les arguments du profil
        cmd.extend(['--epochs', str(config['epochs'])])
        cmd.extend(['--batch_size', str(config['batch_size'])])
        
        if config['models']:
            cmd.extend(['--models'] + config['models'])
        
        if config['datasets']:
            cmd.extend(['--datasets'] + config['datasets'])
        
        # Ajouter les arguments personnalisés
        if custom_args:
            for key, value in custom_args.items():
                if value is not None:
                    cmd.extend([f'--{key}', str(value)])
        
        # Estimation de durée
        duration = self.estimate_duration(profile_name)
        
        print(f"\n🚀 LANCEMENT DU BENCHMARK - PROFIL {profile_name.upper()}")
        print("=" * 60)
        print(f"📊 {config['description']}")
        print(f"⏱️  Durée estimée: {duration}")
        print(f"💻 Commande: {' '.join(cmd)}")
        
        # Demander confirmation pour les longs benchmarks
        if profile_name in ['publication', 'full']:
            response = input(f"\n⚠️  Ce benchmark prendra {duration}. Continuer? (y/N): ")
            if response.lower() != 'y':
                print("❌ Benchmark annulé")
                return False
        
        print(f"\n🔄 Démarrage du benchmark...")
        
        try:
            # Lancer le benchmark
            result = subprocess.run(cmd, cwd=Path.cwd())
            
            if result.returncode == 0:
                print(f"\n✅ Benchmark {profile_name} terminé avec succès!")
                
                # Générer automatiquement les analyses
                print("📊 Génération des analyses...")
                self.generate_analysis()
                
                return True
            else:
                print(f"\n❌ Benchmark {profile_name} échoué (code: {result.returncode})")
                return False
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Benchmark {profile_name} interrompu par l'utilisateur")
            return False
        except Exception as e:
            print(f"\n💥 Erreur lors du benchmark: {e}")
            return False
    
    def generate_analysis(self):
        """Génère automatiquement les analyses et la documentation."""
        
        try:
            # Analyser les résultats
            print("🔍 Analyse des résultats...")
            subprocess.run([sys.executable, 'scripts/analyze_methodology_results.py'], 
                         check=True, cwd=Path.cwd())
            
            # Générer la documentation
            print("📚 Génération de la documentation...")
            subprocess.run([sys.executable, 'scripts/generate_methodology_documentation.py'], 
                         check=True, cwd=Path.cwd())
            
            print("✅ Analyses et documentation générées!")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Erreur lors de la génération d'analyses: {e}")
        except Exception as e:
            print(f"💥 Erreur inattendue: {e}")
    
    def run_custom_benchmark(self, args):
        """Lance un benchmark avec des paramètres personnalisés."""
        
        cmd = [sys.executable, 'scripts/comprehensive_methodology_benchmark.py']
        
        # Ajouter tous les arguments personnalisés
        if args.epochs:
            cmd.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            cmd.extend(['--batch_size', str(args.batch_size)])
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.datasets:
            cmd.extend(['--datasets'] + args.datasets)
        if args.device:
            cmd.extend(['--device', args.device])
        if args.no_augmentation:
            cmd.append('--no-augmentation')
        if args.no_validation:
            cmd.append('--no-validation')
        
        print(f"\n🚀 BENCHMARK PERSONNALISÉ")
        print("=" * 40)
        print(f"💻 Commande: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=Path.cwd())
            
            if result.returncode == 0:
                print("\n✅ Benchmark personnalisé terminé!")
                self.generate_analysis()
                return True
            else:
                print(f"\n❌ Benchmark échoué (code: {result.returncode})")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️  Benchmark interrompu")
            return False

def main():
    parser = argparse.ArgumentParser(description='Lanceur de benchmark méthodologique')
    parser.add_argument('--profile', choices=['quick', 'development', 'validation', 'publication', 'full'],
                       help='Profil de benchmark prédéfini')
    parser.add_argument('--list-profiles', action='store_true',
                       help='Afficher les profils disponibles')
    
    # Arguments pour benchmark personnalisé
    parser.add_argument('--epochs', type=int, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, help='Taille du batch')
    parser.add_argument('--models', nargs='*', help='Modèles à tester')
    parser.add_argument('--datasets', nargs='*', help='Datasets à tester')
    parser.add_argument('--device', help='Device (auto/cpu/cuda)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Désactiver l\'augmentation de données')
    parser.add_argument('--no-validation', action='store_true',
                       help='Désactiver la validation des résultats')
    
    # Actions spéciales
    parser.add_argument('--analyze-only', action='store_true',
                       help='Analyser les résultats existants uniquement')
    parser.add_argument('--docs-only', action='store_true',
                       help='Générer la documentation uniquement')
    
    args = parser.parse_args()
    
    launcher = MethodologyLauncher()
    
    # Afficher les profils
    if args.list_profiles:
        launcher.show_profiles()
        return
    
    # Actions spéciales
    if args.analyze_only:
        print("🔍 Analyse des résultats existants...")
        launcher.generate_analysis()
        return
    
    if args.docs_only:
        print("📚 Génération de la documentation...")
        subprocess.run([sys.executable, 'scripts/generate_methodology_documentation.py'])
        return
    
    # Benchmark avec profil
    if args.profile:
        launcher.launch_benchmark(args.profile)
        return
    
    # Benchmark personnalisé si des arguments sont fournis
    if any([args.epochs, args.batch_size, args.models, args.datasets]):
        launcher.run_custom_benchmark(args)
        return
    
    # Sinon, mode interactif
    print("🎯 LANCEUR DE BENCHMARK MÉTHODOLOGIQUE")
    print("=" * 50)
    print("\nOptions disponibles:")
    print("1. Utiliser un profil prédéfini (--profile PROFIL)")
    print("2. Lancer un benchmark personnalisé (--epochs X --models Y...)")
    print("3. Voir les profils disponibles (--list-profiles)")
    print("4. Analyser les résultats existants (--analyze-only)")
    print("5. Générer la documentation (--docs-only)")
    print("\nUtilisez --help pour voir toutes les options.")

if __name__ == '__main__':
    main()