import os
import pathlib
import pandas as pd
import logging

# ตั้งค่าการแสดงผลสถานะใน Terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class IOManager:
    def __init__(self, root_path):
        """
        จัดการเรื่องการเข้าถึง Disk และเตรียมรายการไฟล์รูปภาพ
        """
        self.root_path = pathlib.Path(root_path)
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    def get_image_tasks(self):
        """
        เลียนแบบ Logic การ Scan ไฟล์ใน ipynb:
        สแกนหาโฟลเดอร์ชั้นใน (Class) และรายชื่อไฟล์ภาพ
        Returns: list of tuples (full_path, class_name, filename)
        """
        tasks = []
        if not self.root_path.exists():
            logging.error(f"❌ ไม่พบ Path ของข้อมูล: {self.root_path}")
            return tasks

        # วนลูปหาโฟลเดอร์ย่อย (เช่น Class_01, Class_02)
        for class_dir in sorted(os.listdir(self.root_path)):
            class_path = self.root_path / class_dir
            
            if class_path.is_dir():
                # วนลูปหาไฟล์ภาพในโฟลเดอร์ Class นั้นๆ
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(self.valid_extensions):
                        full_path = str(class_path / fname)
                        tasks.append((full_path, class_dir, fname))
        
        logging.info(f"🔍 สแกนพบรูปภาพทั้งหมด: {len(tasks)} ไฟล์ จาก {len(os.listdir(self.root_path))} คลาส")
        return tasks

    def ensure_directory(self, folder_path):
        """
        ตรวจสอบและสร้างโฟลเดอร์สำหรับบันทึกผล (ป้องกัน Error ตอน Save)
        """
        path = pathlib.Path(folder_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"📁 สร้างโฟลเดอร์ปลายทาง: {folder_path}")
        return path

    def save_csv(self, df, folder_path, file_name):
        """
        บันทึก DataFrame เป็นไฟล์ CSV (รองรับภาษาไทยใน Excel)
        """
        self.ensure_directory(folder_path)
        save_path = os.path.join(folder_path, file_name)
        
        # ใช้ encoding='utf-8-sig' เพื่อให้เปิดใน Excel แล้วภาษาไม่เพี้ยน
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logging.info(f"💾 บันทึก Master Data เรียบร้อย: {save_path}")