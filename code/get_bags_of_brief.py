"""
Get bags of BRIEF features for image classification
"""
from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
from pybrief import PyBrief
import cv2

def get_bags_of_brief(image_paths, vocab_file='vocab_brief.pkl'):
    """
    Extract BRIEF features and create bag-of-words histograms.
    
    Args:
        image_paths: list of image paths
        vocab_file: path to vocabulary file
        
    Returns:
        image_feats: (N, vocab_size) feature matrix
    """
    with open(vocab_file, 'rb') as handle:
        vocab = pickle.load(handle)

    pyBRIEF = PyBrief()
    # BRIEF solo describe, necesitamos un detector (usamos FAST)
    detector = cv2.FastFeatureDetector_create()
    
    image_feats = []
    
    start_time = time()
    print("Construct bags of BRIEF...")
    
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Procesando imagen {i}/{len(image_paths)}")
            
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
        
        # 1. Detectar keypoints con FAST
        keypoints_cv = detector.detect(img_gray, None)
        
        if keypoints_cv is not None and len(keypoints_cv) > 0:
            # Convertir keypoints de OpenCV a formato numpy (y, x)
            keypoints = np.array([[int(kp.pt[1]), int(kp.pt[0])] for kp in keypoints_cv])
            
            # 2. Describir con BRIEF
            _, descriptors = pyBRIEF.compute(img_gray, keypoints)
        else:
            descriptors = None
        
        if descriptors is None or len(descriptors) == 0:
            # If no descriptors, create zero histogram
            hist_norm = np.zeros(len(vocab))
        else:
            # Convert to float for distance calculation
            descriptors_float = descriptors.astype('float32')
            
            # Find nearest vocabulary word for each descriptor
            dist = distance.cdist(vocab, descriptors_float, metric='euclidean')
            idx = np.argmin(dist, axis=0)
            
            # Create histogram
            hist, bin_edges = np.histogram(idx, bins=len(vocab), range=(0, len(vocab)))
            
            # Normalize histogram
            if np.sum(hist) > 0:
                hist_norm = hist.astype('float32') / np.sum(hist)
            else:
                hist_norm = hist.astype('float32')
        
        image_feats.append(hist_norm)
        
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to construct bags of BRIEF.")
    print(f"Feature matrix shape: {image_feats.shape}")
    
    return image_feats