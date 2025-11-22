"""Build vocabulary using BRIEF descriptors from pyBRIEF"""
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from time import time
from pybrief import PyBrief
import cv2
import multiprocessing
from joblib import Parallel, delayed


def extract_brief_from_image(path, max_descriptors=100):
    """
    Extract BRIEF descriptors from a single image.
    This function will be called in parallel.
    """
    try:
        pyBRIEF = PyBrief()
        detector = cv2.FastFeatureDetector_create()
        
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
            
            if descriptors is not None and len(descriptors) > 0:
                # Convertir descriptores binarios (uint8) a float32 para K-means
                descriptors = descriptors.astype('float32')
                
                # Limitar número de descriptores por imagen
                if len(descriptors) > max_descriptors:
                    indices = np.random.choice(len(descriptors), max_descriptors, replace=False)
                    descriptors = descriptors[indices]
                return descriptors
        return None
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def build_vocabulary_brief(image_paths, vocab_size, max_descriptors_per_image=100):
    """
    Extract BRIEF descriptors from training images and cluster them with kmeans.
    
    Args:
        image_paths: list of training image paths
        vocab_size: number of clusters desired
        max_descriptors_per_image: limit descriptors per image for speed
        
    Returns:
        vocab: cluster centers (vocab_size, descriptor_dim)
    """
    n_jobs = multiprocessing.cpu_count()
    
    print(f"Extract BRIEF features using {n_jobs} cores")
    
    # Parallel processing
    bag_of_features = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(extract_brief_from_image)(path, max_descriptors_per_image) 
        for path in image_paths
    )
    
    # Filter out None values and concatenate
    bag_of_features = [desc for desc in bag_of_features if desc is not None]
    
    if not bag_of_features:
        raise ValueError("No BRIEF descriptors extracted from any image!")
    
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    
    print(f"Compute vocab from {len(bag_of_features)} BRIEF descriptors")
    print(f"Descriptor shape: {bag_of_features.shape}")
    start_time = time()
    
    # Use MiniBatchKMeans for faster clustering
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