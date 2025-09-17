#!/usr/bin/env python3
"""Test script to check model availability."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_model_availability():
    """Test availability of all model types."""
    print("=== Test de Disponibilité des Modèles ===\n")
    
    try:
        from scripts.models.model_factory import ModelFactory
        
        # Get available models
        available = ModelFactory.get_available_models()
        
        print("Statut des modèles:")
        for model, status in available.items():
            symbol = "✓" if status else "✗"
            print(f"  {model}: {symbol}")
        
        print(f"\nModèles disponibles: {sum(available.values())}/{len(available)}")
        
        # Test model list
        print("\nModèles supportés:")
        supported = ModelFactory.list_supported_models()
        for i, model in enumerate(supported[:10]):  # Show first 10
            print(f"  - {model}")
        if len(supported) > 10:
            print(f"  ... et {len(supported) - 10} autres")
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_availability()