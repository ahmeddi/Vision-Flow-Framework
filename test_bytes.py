#!/usr/bin/env python3
"""Byte-level test."""

YOLO_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt',
    'yolov7.pt'
]

def test_bytes():
    target = 'yolo11n.pt'
    
    print(f"Target: {repr(target)}")
    print(f"Target bytes: {target.encode()}")
    print(f"Target len: {len(target)}")
    
    for i, model in enumerate(YOLO_MODELS):
        print(f"{i}: {repr(model)} (len={len(model)}) bytes={model.encode()}")
        if 'yolo11n' in model:
            print(f"  Contains 'yolo11n': True")
            print(f"  Exact match: {model == target}")
            print(f"  Byte match: {model.encode() == target.encode()}")
            # Character by character comparison
            if len(model) == len(target):
                for j, (c1, c2) in enumerate(zip(model, target)):
                    if c1 != c2:
                        print(f"  Difference at position {j}: {repr(c1)} vs {repr(c2)}")
                        print(f"  Char codes: {ord(c1)} vs {ord(c2)}")

if __name__ == "__main__":
    test_bytes()