import os, gc, pathlib, numpy as np, matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from collections import defaultdict
from .utils import analyze_single_image # ดึง worker มาใช้

class PathAnalyzer:
    def __init__(self, root_path, extensions=None):
        self.root_path = pathlib.Path(root_path)
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']

    def run(self, save_path='path_analysis_report.png', show_plot=True):
        """
        ฟังก์ชันหลักในการรันระบบวิเคราะห์สถิติภาพทั้งหมด
        """
        # 1. ค้นหาไฟล์ภาพ
        try:
            image_paths = self._get_image_paths()
        except FileNotFoundError as e:
            print(e)
            return

        # 2. ประมวลผลแบบขนาน (Parallel Processing)
        print(f"🚀 Starting dataset analysis with {self.num_workers} cores...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # ใช้ tqdm แสดง Progress Bar
            raw_results = list(tqdm(
                executor.map(self._analyze_image, image_paths),
                total=len(image_paths),
                desc="Analyzing Images"
            ))

        # 3. คัดกรองและจัดกลุ่มข้อมูลตาม Class
        valid_results = [r for r in raw_results if r is not None]
        if not valid_results:
            print("⚠️ No valid images were processed.")
            return

        # เคลียร์ Memory เบื้องต้น
        del raw_results
        gc.collect()

        class_data = defaultdict(lambda: {"m": [], "sq_m": [], "pixels": []})
        for r in valid_results:
            c = r['class']
            class_data[c]["m"].append(r['mean'])
            class_data[c]["sq_m"].append(r['sq_mean'])
            class_data[c]["pixels"].append(r['pixels'])

        # 4. คำนวณค่าสถิติราย Class และ Global
        classes = sorted(class_data.keys())
        report_data = []

        for cls in classes:
            # คำนวณ Mean ของ Class
            cls_mean = np.mean(class_data[cls]["m"], axis=0)
            # คำนวณ Std ของ Class (ใช้สูตร: sqrt(E[X^2] - (E[X])^2))
            cls_std = np.sqrt(np.mean(class_data[cls]["sq_m"], axis=0) - cls_mean**2)
            report_data.append((cls, cls_mean, cls_std))

        # 5. การแสดงผล (Visualization & Terminal Stats)
        if show_plot:
            self._visualize(class_data, report_data, save_path)
        
        self._print_stats(report_data)

        # 6. คืนทรัพยากร (Final Cleanup)
        del class_data
        gc.collect()
        print(f"✨ Process finished. Total images analyzed: {len(valid_results)}")

    def _visualize(self, class_data, report_data, save_path):
        """จัดการเรื่องการวาดกราฟ Histogram ของแต่ละ Class และบันทึกรูป"""
        classes = sorted(class_data.keys())
        n_classes = len(classes)
        
        # สร้าง Canvas สำหรับวาดกราฟ (Row = จำนวน Class, Column = 3 สี R,G,B)
        fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes), squeeze=False)
        plot_colors = {'Red': '#FF0000', 'Green': '#48FF48', 'Blue': '#0080FF'}

        print(f"📊 Generating distribution plots...")
        for row_i, cls in enumerate(classes):
            # รวม pixels ที่สุ่มมาทั้งหมดของ class นี้
            all_pixels = np.vstack(class_data[cls]["pixels"])
            
            for col_j, (ch_name, ch_color) in enumerate(plot_colors.items()):
                ax = axes[row_i, col_j]
                # วาด Histogram (คูณ 255 เพื่อกลับเป็นค่าสีปกติ)
                ax.hist(all_pixels[:, col_j] * 255, bins=50, color=ch_color, alpha=0.7, density=True)
                
                # ตกแต่งกราฟ
                ax.set_title(f"Class: {cls} | Channel: {ch_name}", fontsize=10)
                ax.set_xlim(0, 255)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                if row_i == n_classes - 1:
                    ax.set_xlabel("Pixel Value")
                if col_j == 0:
                    ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"💾 Report saved successfully at: {save_path}")
        plt.close(fig) # ปิดเพื่อคืน Memory
        plt.savefig(save_path)
        print(f"📊 Report saved to {save_path}")

    def _print_stats(self, report_data):
        """แสดงผลสถิติสรุปบน Terminal สำหรับใช้ทำ Normalization"""
        print("\n" + "═"*85)
        print(f"║ {'DATASET ANALYSIS SUMMARY':^81} ║")
        print("═"*85)
        print(f"  {'Class (Folder)':<25} {'Mean (R, G, B)':<28} {'Std (R, G, B)':<28}")
        print("─"*85)
        
        all_m, all_s = [], []
        for cls, m, s in report_data:
            m_str = f"[{', '.join([f'{x:.4f}' for x in m])}]"
            s_str = f"[{', '.join([f'{x:.4f}' for x in s])}]"
            print(f"  {cls:<25} {m_str:<28} {s_str:<28}")
            all_m.append(m)
            all_s.append(s)

        # คำนวณค่าเฉลี่ยรวมทั้ง Dataset
        g_m = np.mean(all_m, axis=0)
        g_s = np.mean(all_s, axis=0)
        
        print("─"*85)
        print(f"  {'GLOBAL DATASET MEAN':<25} {str(g_m.round(6)):<28}")
        print(f"  {'GLOBAL DATASET STD':<25} {str(g_s.round(6)):<28}")
        print("═"*85)
        
        # สรุปบรรทัดที่ก๊อปปี้ไปวางในโค้ด PyTorch/TensorFlow ได้เลย
        print(f"\n💡 For PyTorch transforms:")
        print(f"   transforms.Normalize(mean={list(g_m.round(6))}, std={list(g_s.round(6))})")
        print("\n✅ Analysis Complete.\n")
    