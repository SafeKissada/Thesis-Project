import os

try:
    from pipeline_system.io_manager import IOManager
    from pipeline_system.exporter import run_pipeline
except ImportError as e:
    print(f"❌ Import Error: {e}")

def main():
    # 1. ตั้งค่า Path 
    DATA_PATH = r"D:\0PROJECT\thesis\data\raw"
    SAVE_PATH = r"D:\0PROJECT\thesis\docs\saved"

    # 2. เตรียมรายการงาน
    io = IOManager(DATA_PATH)
    tasks = io.get_image_tasks() 
    
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