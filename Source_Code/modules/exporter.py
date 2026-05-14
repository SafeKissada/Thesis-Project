import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import pandas as pd
from .processor import process_single_image

def run_pipeline(tasks):
    """
    รับรายการไฟล์มาประมวลผลแบบขนานและส่งคืนเป็น DataFrame
    """
    results = []
    # ใช้จำนวน CPU ทั้งหมดในเครื่อง Windows ของคุณ
    cpus = 2
    
    print(f"🚀 Starting parallel processing with {cpus} cores...")
    
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        # ส่งงานเข้าประมวลผล
        future_to_task = {executor.submit(process_single_image, t): t for t in tasks}
        
        # แสดงผลความคืบหน้าด้วย tqdm
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing Images"):
            results.append(future.result())

    return pd.DataFrame(results)