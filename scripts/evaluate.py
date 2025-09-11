"""Evaluation script to compute metrics and latency for trained YOLO models."""
import argparse, json, time, statistics
from pathlib import Path
import torch
from ultralytics import YOLO

@torch.inference_mode()
def benchmark_latency(model, imgsz=640, device='cuda', warmup=10, runs=50):
    import torch
    dummy = torch.randn(1,3,imgsz,imgsz, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model(dummy)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        times.append(time.time()-start)
    avg = sum(times)/len(times)
    p95 = sorted(times)[int(0.95*len(times))-1]
    fps = 1/avg if avg>0 else 0
    return {'latency_ms': avg*1000, 'p95_ms': p95*1000, 'fps': fps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', default='results/runs')
    ap.add_argument('--pattern', default='*')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--output', default='results/eval_summary.json')
    args = ap.parse_args()

    models = []
    for p in Path(args.models_dir).glob(args.pattern):
        if p.is_dir():
            best = p / 'weights' / 'best.pt'
            if best.exists():
                models.append(best)
    all_results = []
    for m in models:
        print(f"Evaluating {m}")
        y = YOLO(str(m))
        metrics = y.val()  # uses data inside training config path
        lat = benchmark_latency(y.model, imgsz=args.imgsz, device=args.device)
        rd = metrics.results_dict
        rd.update(lat)
        rd['model_path'] = str(m)
        all_results.append(rd)
    with open(args.output,'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {args.output}")

if __name__ == '__main__':
    main()
