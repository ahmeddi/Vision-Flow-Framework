"""Unified training script wrapper leveraging Ultralytics YOLO for multiple models.
This is a simplified orchestrator; extend for full reproducibility and logging.
"""
import argparse, yaml, os, json, time
from pathlib import Path
from ultralytics import YOLO

MODELS = [
    'yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt',
    'yolov11n.pt','yolov11s.pt','yolov11m.pt','yolov11l.pt','yolov11x.pt'
]

# Placeholder for additional models (require integration): YOLO-NAS, YOLOX, PP-YOLOE, EfficientDet, DETR, RT-DETR
# Those need separate training codebases; here we focus on YOLO variants for now.

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/base.yaml')
    ap.add_argument('--data', required=True, help='Path to YOLO data YAML defining train/val sets')
    ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--output', default='results')
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    results_summary = []
    for model_name in args.models:
        print(f"=== Training {model_name} ===")
        ymodel = YOLO(model_name)
        start = time.time()
        r = ymodel.train(
            data=args.data,
            epochs=cfg.get('epochs', 50),
            imgsz=cfg.get('img_size', 640),
            batch=cfg.get('batch_size', 16),
            seed=cfg.get('seed', 42),
            optimizer=cfg.get('optimizer','SGD'),
            lr0=cfg.get('learning_rate', 0.01),
            momentum=cfg.get('momentum', 0.937),
            weight_decay=cfg.get('weight_decay', 0.0005),
            project=str(out_root / 'runs'),
            name=model_name.replace('.pt',''),
            exist_ok=True
        )
        elapsed = time.time() - start
        metrics = r.results_dict
        metrics['model'] = model_name
        metrics['train_time_s'] = elapsed
        results_summary.append(metrics)

    with open(out_root / 'training_summary.json','w') as f:
        json.dump(results_summary, f, indent=2)
    print('Saved summary to training_summary.json')

if __name__ == '__main__':
    main()
