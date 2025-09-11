"""Energy consumption measurement for YOLO models.
Logs GPU power consumption during inference using NVML or psutil for CPU.
"""
import argparse, time, json, statistics, threading
from pathlib import Path
import torch
from ultralytics import YOLO

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False

import psutil

class EnergyLogger:
    def __init__(self, device='cuda', interval=1.0):
        self.device = device
        self.interval = interval
        self.power_samples = []
        self.logging = False
        
        if device.startswith('cuda') and NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.use_gpu = True
            except:
                self.use_gpu = False
        else:
            self.use_gpu = False
    
    def start_logging(self):
        self.logging = True
        self.power_samples = []
        self.thread = threading.Thread(target=self._log_power)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_logging(self):
        self.logging = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        return self.power_samples
    
    def _log_power(self):
        while self.logging:
            try:
                if self.use_gpu:
                    # GPU power in milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_w = power_mw / 1000.0
                else:
                    # CPU power estimation (very rough)
                    cpu_percent = psutil.cpu_percent(interval=None)
                    # Assume ~65W TDP for typical CPU, scale by usage
                    power_w = 65.0 * (cpu_percent / 100.0)
                
                self.power_samples.append({
                    'timestamp': time.time(),
                    'power_w': power_w
                })
            except Exception as e:
                print(f"Power logging error: {e}")
                
            time.sleep(self.interval)

@torch.inference_mode()
def measure_energy_consumption(model_path, device='cuda', n_images=100, batch_size=1, img_size=640):
    """Measure energy consumption for model inference."""
    
    model = YOLO(model_path)
    model.to(device)
    
    # Create dummy data
    dummy_batch = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = model(dummy_batch)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # Start energy logging
    logger = EnergyLogger(device=device, interval=0.5)
    logger.start_logging()
    
    start_time = time.time()
    
    # Run inference
    print(f"Running {n_images} inferences...")
    for i in range(n_images):
        _ = model(dummy_batch)
        if i % 20 == 0:
            print(f"Progress: {i}/{n_images}")
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Stop logging
    power_samples = logger.stop_logging()
    
    # Calculate metrics
    total_time = end_time - start_time
    fps = n_images / total_time
    
    if len(power_samples) > 1:
        powers = [s['power_w'] for s in power_samples]
        avg_power = statistics.mean(powers)
        total_energy = avg_power * total_time  # Joules (W * s)
        energy_per_image = total_energy / n_images
    else:
        avg_power = 0
        total_energy = 0
        energy_per_image = 0
    
    results = {
        'model_path': str(model_path),
        'device': device,
        'n_images': n_images,
        'total_time_s': total_time,
        'fps': fps,
        'avg_power_w': avg_power,
        'total_energy_j': total_energy,
        'energy_per_image_j': energy_per_image,
        'power_samples': len(power_samples)
    }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Model paths to benchmark')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--n_images', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--output', default='results/energy_results.json')
    args = parser.parse_args()
    
    all_results = []
    
    for model_path in args.models:
        print(f"\\n=== Measuring {model_path} ===")
        result = measure_energy_consumption(
            model_path, args.device, args.n_images, 
            args.batch_size, args.img_size
        )
        all_results.append(result)
        
        print(f"FPS: {result['fps']:.2f}")
        print(f"Avg Power: {result['avg_power_w']:.2f}W")
        print(f"Energy/Image: {result['energy_per_image_j']:.4f}J")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
