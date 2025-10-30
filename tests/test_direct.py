#!/usr/bin/env python3
"""Direct test without import."""

# Define models directly
YOLO_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt',
    'yolov7.pt'
]

def test_direct():
    target = 'yolo11n.pt'
    print(f"Target: {repr(target)}")
    print(f"YOLO_MODELS: {YOLO_MODELS}")
    print(f"Target in list: {target in YOLO_MODELS}")
    
    # Manual search
    for i, model in enumerate(YOLO_MODELS):
        if model == target:
            print(f"Found exact match at index {i}: {repr(model)}")
            return True
    
    print("No exact match found")
    return False

if __name__ == "__main__":
    test_direct()