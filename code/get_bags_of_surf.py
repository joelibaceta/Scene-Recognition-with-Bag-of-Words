"""
Get bags of SURF features for image classification
"""
from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
from pysurf import PySurf

def get_bags_of_surf(image_paths, vocab_file='vocab_surf.pkl'):
    """
    Extract SURF features and create bag-of-words histograms.
    
    Args:
        image_paths: list of image paths
        vocab_file: path to vocabulary file
        
    Returns:
        image_feats: (N, vocab_size) feature matrix
    """
    with open(vocab_file, 'rb') as handle:
        vocab = pickle.load(handle)
    
    image_feats = []

    # Use lower threshold for better detection in low-contrast images
    pySURF = PySurf(
        hessian_thresh=0.0001,  # Reduced from default 0.004
        n_scales=5,
        edge_threshold=20.0,
        upright=False
    )
    
    start_time = time()
    print("Construct bags of SURF...")
    
    for path in image_paths:
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
            
        # Extract SURF descriptors
        keypoints, descriptors = pySURF.detect_and_describe(img_gray)
        
        if descriptors is None or len(descriptors) == 0:
            # If no descriptors, create zero histogram
            hist_norm = np.zeros(len(vocab))
        else:
            # Find nearest vocabulary word for each descriptor
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
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
    print(f"It takes {end_time - start_time:.2f} seconds to construct bags of SURF.")
    
    return image_feats
