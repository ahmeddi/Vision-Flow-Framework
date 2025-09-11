"""Generate a tiny dummy dataset for quick script smoke tests."""
from pathlib import Path
import argparse, random
from PIL import Image, ImageDraw

def gen_image(path, label_path, idx):
    w, h = 640, 640
    img = Image.new('RGB', (w,h), (30,120,30))
    draw = ImageDraw.Draw(img)
    # random box
    bw, bh = random.randint(60,200), random.randint(60,200)
    x0 = random.randint(0, w-bw)
    y0 = random.randint(0, h-bh)
    x1, y1 = x0+bw, y0+bh
    draw.rectangle([x0,y0,x1,y1], outline='yellow', width=3)
    img.save(path / f"img_{idx}.jpg")
    # YOLO txt: class cx cy bw bh normalized
    cx = (x0 + x1)/2 / w
    cy = (y0 + y1)/2 / h
    nw = (x1 - x0)/ w
    nh = (y1 - y0)/ h
    (label_path / f"img_{idx}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/dummy')
    ap.add_argument('--n_train', type=int, default=8)
    ap.add_argument('--n_val', type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    for split, n in [('train', args.n_train), ('val', args.n_val)]:
        img_dir = root / 'images' / split
        lab_dir = root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            gen_image(img_dir, lab_dir, i)
    print('Dummy dataset generated.')

if __name__ == '__main__':
    main()
