"""Build vocabulary using SURF descriptors from pySURF"""
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from time import time
from pysurf import PySurf
import multiprocessing
from joblib import Parallel, delayed

def extract_surf_from_image(path, max_descriptors=100):
    """
    Extract SURF descriptors from a single image.
    This function will be called in parallel.
    """
    try:
        # Initialize PySurf for this worker
        # Use lower threshold for better detection in low-contrast images
        surf = PySurf(
            hessian_thresh=0.0001,  # Reduced from 0.0005
            n_scales=5,
            edge_threshold=20.0,
            upright=False
        )
        
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
            
        # Extract SURF
        keypoints, descriptors = surf.detect_and_describe(img_gray)
        
        if descriptors is not None and len(descriptors) > 0:
            # Limit descriptors per image
            if len(descriptors) > max_descriptors:
                indices = np.random.choice(len(descriptors), max_descriptors, replace=False)
                descriptors = descriptors[indices]
            return descriptors
        else:
            return None
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def build_vocabulary_surf(image_paths, vocab_size, max_descriptors_per_image=100):
    """
    Extract SURF descriptors from training images and cluster them with kmeans.
    
    Args:
        image_paths: list of training image paths
        vocab_size: number of clusters desired
        max_descriptors_per_image: limit descriptors per image for speed
        
    Returns:
        vocab: cluster centers (vocab_size, descriptor_dim)
    """
    bag_of_features = []
    # Note: PySurf instances are created in worker processes
    n_jobs = multiprocessing.cpu_count()

    print(f"Extract SURF features usando {n_jobs} cores")
    
    bag_of_features = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(extract_surf_from_image)(path, max_descriptors_per_image) 
        for path in image_paths
    )
    
    
    # Concatenate all descriptors
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    
    print(f"Compute vocab from {len(bag_of_features)} SURF descriptors")
    start_time = time()
    
    # Use MiniBatchKMeans for faster clustering with parallel processing
    kmeans_model = MiniBatchKMeans(
        n_clusters=vocab_size,
        init='k-means++',
        batch_size=1000,
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose=1,
        compute_labels=False 
    )
    
    kmeans_model.fit(bag_of_features)
    vocab = kmeans_model.cluster_centers_
    
    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to compute vocab.")
    print(f"Vocabulario shape: {vocab.shape}")
    
    return vocab