import os, gc, pathlib, numpy as np, matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from collections import defaultdict
from utils import analyze_single_image # ดึง worker มาใช้

class PathAnalyzer:
    def __init__(self, root_path, extensions=None):
        self.root_path = pathlib.Path(root_path)
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']

    def run(self, save_path='report.png', show_plot=True):
        all_files = []
        for ext in self.extensions:
            all_files.extend(list(self.root_path.rglob(f"*{ext}")))
            all_files.extend(list(self.root_path.rglob(f"*{ext.upper()}")))

        print(f"🚀 Processing {len(all_files)} images...")
        with ProcessPoolExecutor() as executor:
            raw_results = list(tqdm(executor.map(analyze_single_image, all_files), total=len(all_files)))

        valid_results = [r for r in raw_results if r is not None]
        class_data = defaultdict(lambda: {"m": [], "sq_m": [], "pixels": []})
        for r in valid_results:
            class_data[r['class']]["m"].append(r['mean'])
            class_data[r['class']]["sq_m"].append(r['sq_mean'])
            class_data[r['class']]["pixels"].append(r['pixels'])

        # สรุปตัวเลข
        report_data = []
        for cls in sorted(class_data.keys()):
            m = np.mean(class_data[cls]["m"], axis=0)
            s = np.sqrt(np.mean(class_data[cls]["sq_m"], axis=0) - m**2)
            report_data.append((cls, m, s))

        if show_plot:
            self._visualize(class_data, report_data, save_path)
        
        self._print_stats(report_data)

    def _visualize(self, class_data, report_data, save_path):
        # ... (โค้ดวาดกราฟเดิมของคุณ) ...
        plt.savefig(save_path)
        print(f"📊 Report saved to {save_path}")

    def _print_stats(self, report_data):
        # ... (โค้ด Print สถิติเดิมของคุณ) ...
        print("✅ Analysis Complete.")