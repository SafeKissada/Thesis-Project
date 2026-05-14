import os
from modules import IOManager, run_pipeline

def main():
    # 1. ตั้งค่า Path (ใช้ r นำหน้าเพื่อป้องกันปัญหาเครื่องหมาย \)
    DATA_PATH = r"D:\0PROJECT\thesis\data\raw"
    SAVE_PATH = r"D:\0PROJECT\thesis\docs\saved"

    # 2. เตรียมรายการงาน
    io = IOManager(DATA_PATH)
    tasks = io.get_image_tasks() # ดึงรายการรูป (path, class, name)
    
    if tasks:
        # 3. รัน Pipeline (สกัด Features)
        df = run_pipeline(tasks)
        
        # 4. บันทึกผล
        io.save_csv(df, SAVE_PATH, "Metadata_MasterData.csv")
    else:
        print("❌ No images found in the specified path.")

if __name__ == "__main__":
    # สำคัญมากสำหรับ Windows: ต้องรันผ่านฟังก์ชัน main() ภายใต้เงื่อนไขนี้
    main()