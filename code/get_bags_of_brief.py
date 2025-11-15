"""
Get bags of BRIEF features for image classification
"""
from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
from time import time
import pyBRIEF

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
    
    image_feats = []
    
    start_time = time()
    print("Construct bags of BRIEF...")
    
    for path in image_paths:
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
            
        # Extract BRIEF descriptors
        keypoints, descriptors = pyBRIEF.detectAndCompute(img_gray)
        
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
    
    return image_feats
