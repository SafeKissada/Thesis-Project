import numpy as np
from PIL import Image

def analyze_single_image(file_path):
    try:
        class_name = file_path.parent.name
        with Image.open(file_path) as img:
            img_arr = np.array(img.convert('RGB')) / 255.0
            m = np.mean(img_arr, axis=(0, 1))
            s = np.mean(img_arr**2, axis=(0, 1))
            
            flat_pixels = img_arr.reshape(-1, 3)
            sample_idx = np.random.choice(flat_pixels.shape[0], min(1000, flat_pixels.shape[0]), replace=False)
            return {"class": class_name, "mean": m, "sq_mean": s, "pixels": flat_pixels[sample_idx]}
    except:
        return None