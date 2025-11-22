from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
# Usar OpenCV en lugar de cyvlfeat
from cv2_sift_utils import dsift
from time import time
from joblib import Parallel, delayed
import multiprocessing

def get_bags_of_sifts(image_paths, vocab_file='vocab_sift.pkl'):
    """
    Extract SIFT features and create bag-of-words histograms.
    
    Args:
        image_paths: list of image paths
        vocab_file: path to vocabulary file
        
    Returns:
        image_feats: (N, vocab_size) feature matrix
    """
    
    with open(vocab_file, 'rb') as handle:
        vocab = pickle.load(handle)
    
    # Función auxiliar para procesar una imagen
    def process_image(path, vocab):
        try:
            img = np.asarray(Image.open(path), dtype='float32')
            # Usar step más grande para ser más rápido (5,5 en lugar de 1,1)
            frames, descriptors = dsift(img, step=[5,5], fast=True)
            
            if descriptors is None or len(descriptors) == 0:
                return np.zeros(len(vocab), dtype='float32')
            
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
            idx = np.argmin(dist, axis=0)
            hist, _ = np.histogram(idx, bins=len(vocab), range=(0, len(vocab)))
            
            # Normalize histogram
            if np.sum(hist) > 0:
                return hist.astype('float32') / np.sum(hist)
            else:
                return hist.astype('float32')
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return np.zeros(len(vocab), dtype='float32')
    
    start_time = time()
    print("Construct bags of SIFT (parallelized)...")
    
    # Procesar en paralelo
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} CPU cores")
    
    image_feats = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(process_image)(path, vocab) 
        for path in image_paths
    )
    
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print(f"Bag of SIFT construction took {end_time - start_time:.2f}s")
    
    return image_feats