"""Get bags of SURF features for image classification"""
from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
from pysurf import PySurf
import multiprocessing
from joblib import Parallel, delayed

def extract_surf_histogram(path, vocab):
    """
    Extract SURF features and compute histogram for a single image.
    This function will be called in parallel.
    """
    try:
        # Initialize PySurf for this worker
        pySURF = PySurf(
            hessian_thresh=0.0001,
            n_scales=5,
            edge_threshold=20.0,
            upright=False
        )
        
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
            return np.zeros(len(vocab), dtype='float32')
        else:
            # Find nearest vocabulary word for each descriptor
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
            idx = np.argmin(dist, axis=0)
            
            # Create histogram
            hist, _ = np.histogram(idx, bins=len(vocab), range=(0, len(vocab)))
            
            # Normalize histogram
            if np.sum(hist) > 0:
                return hist.astype('float32') / np.sum(hist)
            else:
                return hist.astype('float32')
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros(len(vocab), dtype='float32')

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
    
    n_jobs = multiprocessing.cpu_count()
    
    start_time = time()
    print(f"Construct bags of SURF using {n_jobs} cores...")
    
    # Parallel processing
    image_feats = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(extract_surf_histogram)(path, vocab) 
        for path in image_paths
    )
    
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to construct bags of SURF.")
    
    return image_feats
