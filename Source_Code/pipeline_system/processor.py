import os 
import hashlib
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew  

def process_single_image(args):
    """
    รับ Tuple (path, class_name, filename) มาประมวลผลสกัด Features
    """
    fpath, class_name, fname = args
    ext = os.path.splitext(fname)[1].lower()
    
    # พื้นฐาน Metadata
    record = {
        'filepath': os.path.abspath(fpath),
        'class_name': class_name,
        'filename': fname,
        'ext': ext.replace('.', ''),
        'file_size': os.path.getsize(fpath),
        "is_corrupt": False
    }

    try:
        # 1. MD5 Hash (Read as Binary)
        hash_md5 = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        record['md5_hash'] = hash_md5.hexdigest()
        
        # 2. Load Image with OpenCV
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            raise ValueError('OpenCV cannot read image')
        
        # Image Metadata
        h, w, _ = img_bgr.shape
        record.update({
            'width': w,
            'height': h,
            'pixel' : w * h,
            'aspect_ratio': round(w / h, 8)
        })

        # 3. Preprocessing
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Color & Brightness
        r_m, g_m, b_m = np.mean(img_rgb, axis=(0, 1))
        r_s, g_s, b_s = np.std(img_rgb, axis=(0, 1))
        
        # Luma calculation (ITU-R 601)
        lum_array = (0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2])

        # Feature Scoring
        blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        edge_score = np.mean(cv2.Canny(img_gray, 100, 200))
        
        # Texture (GLCM) - แนะนำให้ลด levels หากต้องการความเร็ว
        glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        record.update({
            'red_mean': round(float(r_m), 6),
            'green_mean': round(float(g_m), 6),
            'blue_mean': round(float(b_m), 6),
            'red_std': round(float(r_s), 6), 
            'green_std': round(float(g_s), 6), 
            'blue_std': round(float(b_s), 6),
            'contrast': round(float(lum_array.std()), 6),
            'luma_skewness': round(float(skew(lum_array.flatten())), 6),
            'brightness': round(float(lum_array.mean()), 6),
            'blur_score': round(float(blur_score), 6),
            'edge_score': round(float(edge_score), 6),
            'texture_contrast': round(float(graycoprops(glcm, 'contrast')[0, 0]), 6),
            'texture_homogeneity': round(float(graycoprops(glcm, 'homogeneity')[0, 0]), 6),
            'entropy': round(float(shannon_entropy(img_gray)), 6)
        })
    
    except Exception as e:
        record['is_corrupt'] = True
        record['error_log'] = str(e)

    return record