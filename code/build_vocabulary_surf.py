"""
Build vocabulary using SURF descriptors from pySURF
"""
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from time import time
from pysurf import PySurf
import multiprocessing


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
    pySURF = PySurf()
    n_jobs = multiprocessing.cpu_count()

    print(f"Extract SURF features usando {n_jobs} cores")
    
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Procesando imagen {i}/{len(image_paths)}")
            
        img = np.asarray(Image.open(path), dtype='uint8')
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')
        else:
            img_gray = img
            
        # Extract SURF descriptors
        keypoints, descriptors = pySURF.detect_and_describe(img_gray)
        
        if descriptors is not None and len(descriptors) > 0:
            # Limitar número de descriptores por imagen
            if len(descriptors) > max_descriptors_per_image:
                indices = np.random.choice(len(descriptors), max_descriptors_per_image, replace=False)
                descriptors = descriptors[indices]
            bag_of_features.append(descriptors)
    
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