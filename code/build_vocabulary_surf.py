"""
Build vocabulary using SURF descriptors from pySURF
"""
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from time import time
from pysurf import PySurf

def build_vocabulary_surf(image_paths, vocab_size):
    """
    Extract SURF descriptors from training images and cluster them with kmeans.
    
    Args:
        image_paths: list of training image paths
        vocab_size: number of clusters desired
        
    Returns:
        vocab: cluster centers (vocab_size, descriptor_dim)
    """
    bag_of_features = []
    
    print("Extract SURF features")
    
    for path in image_paths:
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
            
        # Extract SURF descriptors
        keypoints, descriptors = pySURF.detectAndCompute(img_gray)
        
        if descriptors is not None and len(descriptors) > 0:
            bag_of_features.append(descriptors)
    
    # Concatenate all descriptors
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    
    print(f"Compute vocab from {len(bag_of_features)} SURF descriptors")
    start_time = time()
    
    # Use KMeans from scikit-learn
    kmeans_model = KMeans(
        n_clusters=vocab_size,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose=0
    )
    
    kmeans_model.fit(bag_of_features)
    vocab = kmeans_model.cluster_centers_
    
    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to compute vocab.")
    
    return vocab
