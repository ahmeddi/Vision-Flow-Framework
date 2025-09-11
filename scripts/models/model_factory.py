"""
Model Factory for creating different model architectures.
Provides a unified interface for model creation.
"""

from pathlib import Path
from typing import Optional, Dict, Any

# Import model wrappers with fallback handling
try:
    from .yolo_wrapper import YOLOWrapper
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO models not available")

try:
    from .yolo_nas_wrapper import YOLONASWrapper
    YOLO_NAS_AVAILABLE = True
except ImportError:
    YOLO_NAS_AVAILABLE = False
    print("Warning: YOLO-NAS models not available")

try:
    from .yolox_wrapper import YOLOXWrapper
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False
    print("Warning: YOLOX models not available")

try:
    from .efficientdet_wrapper import EfficientDetWrapper
    EFFICIENTDET_AVAILABLE = True
except ImportError:
    EFFICIENTDET_AVAILABLE = False
    print("Warning: EfficientDet models not available")

try:
    from .detr_wrapper import DETRWrapper
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Warning: DETR models not available")

try:
    from .yolov7_wrapper import YOLOv7Wrapper
    YOLOV7_AVAILABLE = True
except ImportError:
    YOLOV7_AVAILABLE = False
    print("Warning: YOLOv7 models not available")

try:
    from .pp_yoloe_wrapper import PPYOLOEWrapper
    PP_YOLOE_AVAILABLE = True
except ImportError:
    PP_YOLOE_AVAILABLE = False
    print("Warning: PP-YOLOE models not available")


def detect_model_type(model_name: str) -> str:
    """
    Detect model type from model name or path.
    
    Args:
        model_name: Model name, path, or identifier
        
    Returns:
        Model type string
    """
    model_name = model_name.lower()
    
    if any(x in model_name for x in ['yolov8', 'yolov11']):
        return 'yolo'
    elif 'yolov7' in model_name:
        return 'yolov7'
    elif 'pp_yoloe' in model_name or 'pp-yoloe' in model_name:
        return 'pp_yoloe'
    elif 'yolo_nas' in model_name or 'yolo-nas' in model_name:
        return 'yolo_nas'
    elif 'yolox' in model_name:
        return 'yolox'
    elif 'efficientdet' in model_name or 'effdet' in model_name:
        return 'efficientdet'
    elif any(x in model_name for x in ['detr', 'rt_detr', 'rt-detr']):
        return 'detr'
    elif 'yolo' in model_name:  # Generic YOLO fallback
        return 'yolo'
    else:
        return 'yolo'  # Default fallback


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(model_name: str, model_type: Optional[str] = None, **kwargs):
        """
        Create a model wrapper based on model name and type.
        
        Args:
            model_name: Name or path of the model
            model_type: Explicit model type (optional, will auto-detect if None)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model wrapper instance
            
        Raises:
            ValueError: If model type is unsupported or unavailable
        """
        if model_type is None:
            model_type = detect_model_type(model_name)
        
        if model_type == 'yolo':
            if not YOLO_AVAILABLE:
                raise ValueError("YOLO models not available. Install ultralytics.")
            return YOLOWrapper(model_name, **kwargs)
        
        elif model_type == 'yolov7':
            if not YOLOV7_AVAILABLE:
                raise ValueError("YOLOv7 models not available.")
            return YOLOv7Wrapper(model_name, **kwargs)
        
        elif model_type == 'pp_yoloe':
            if not PP_YOLOE_AVAILABLE:
                raise ValueError("PP-YOLOE models not available. Install PaddleDetection.")
            return PPYOLOEWrapper(model_name, **kwargs)
        
        elif model_type == 'yolo_nas':
            if not YOLO_NAS_AVAILABLE:
                raise ValueError("YOLO-NAS models not available. Install super-gradients.")
            return YOLONASWrapper(model_name, **kwargs)
        
        elif model_type == 'yolox':
            if not YOLOX_AVAILABLE:
                raise ValueError("YOLOX models not available. Install yolox.")
            return YOLOXWrapper(model_name, **kwargs)
        
        elif model_type == 'efficientdet':
            if not EFFICIENTDET_AVAILABLE:
                raise ValueError("EfficientDet models not available. Install effdet and timm.")
            return EfficientDetWrapper(model_name, **kwargs)
        
        elif model_type == 'detr':
            if not DETR_AVAILABLE:
                raise ValueError("DETR models not available. Install transformers.")
            return DETRWrapper(model_name, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, bool]:
        """
        Get dictionary of available model types.
        
        Returns:
            Dictionary mapping model types to availability
        """
        return {
            'yolo': YOLO_AVAILABLE,
            'yolov7': YOLOV7_AVAILABLE,
            'pp_yoloe': PP_YOLOE_AVAILABLE,
            'yolo_nas': YOLO_NAS_AVAILABLE,
            'yolox': YOLOX_AVAILABLE,
            'efficientdet': EFFICIENTDET_AVAILABLE,
            'detr': DETR_AVAILABLE
        }
    
    @staticmethod
    def list_supported_models() -> list:
        """
        List all supported model architectures.
        
        Returns:
            List of supported model names/patterns
        """
        models = []
        
        if YOLO_AVAILABLE:
            models.extend([
                'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt'
            ])
        
        if YOLOV7_AVAILABLE:
            models.extend(['yolov7.pt', 'yolov7-tiny.pt', 'yolov7x.pt'])
        
        if PP_YOLOE_AVAILABLE:
            models.extend(['pp_yoloe_s', 'pp_yoloe_m', 'pp_yoloe_l', 'pp_yoloe_x'])
        
        if YOLO_NAS_AVAILABLE:
            models.extend(['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'])
        
        if YOLOX_AVAILABLE:
            models.extend(['yolox-nano', 'yolox-tiny', 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x'])
        
        if EFFICIENTDET_AVAILABLE:
            models.extend([f'efficientdet-d{i}' for i in range(8)])
        
        if DETR_AVAILABLE:
            models.extend(['detr-resnet-50', 'detr-resnet-101', 'rt-detr-l', 'rt-detr-x'])
        
        return models


# Convenience functions
def create_model(model_name: str, model_type: Optional[str] = None, **kwargs):
    """Convenience function to create a model."""
    return ModelFactory.create_model(model_name, model_type, **kwargs)


def get_available_models() -> Dict[str, bool]:
    """Convenience function to get available models."""
    return ModelFactory.get_available_models()


def list_supported_models() -> list:
    """Convenience function to list supported models."""
    return ModelFactory.list_supported_models()


if __name__ == "__main__":
    # Test the factory
    print("Model Factory Test")
    print("=" * 50)
    
    available = get_available_models()
    print("Available model types:")
    for model_type, is_available in available.items():
        status = "✅" if is_available else "❌"
        print(f"  {status} {model_type}")
    
    print(f"\nSupported models: {len(list_supported_models())}")
    
    # Test creating a YOLO model if available
    if available['yolo']:
        try:
            model = create_model('yolov8n.pt')
            print(f"✅ Successfully created YOLO model: {model.model_name}")
        except Exception as e:
            print(f"❌ Failed to create YOLO model: {e}")
