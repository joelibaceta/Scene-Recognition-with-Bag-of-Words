from PIL import Image
import numpy as np
# Usar OpenCV en lugar de cyvlfeat
from cv2_sift_utils import dsift, kmeans
from time import time
from joblib import Parallel, delayed
import multiprocessing


def build_vocabulary_sift(image_paths, vocab_size):
    """
    Extract SIFT descriptors from training images and cluster them with kmeans.
    
    Args:
        image_paths: list of training image paths
        vocab_size: number of clusters desired
        
    Returns:
        vocab: cluster centers (vocab_size, descriptor_dim)
    """

    '''   
    bag_of_features = []
    print("Extract SIFT features")
    pdb.set_trace()
    for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=[5,5], fast=True)
        bag_of_features.append(descriptors)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    pdb.set_trace()
    print("Compute vocab")
    start_time = time()
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    end_time = time()
    print("It takes ", (start_time - end_time), " to compute vocab.")
    '''
    
    # Función auxiliar para procesar una imagen
    def extract_features_from_image(path, step_size=15):
        try:
            img = np.asarray(Image.open(path), dtype='float32')
            frames, descriptors = dsift(img, step=[step_size, step_size], fast=True)
            return descriptors
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return np.array([], dtype='float32').reshape(0, 128)
    
    bag_of_features = []
    
    print("Extract SIFT features (parallelized)")
    step_size = 15
    
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} CPU cores")
    
    start_extract = time()
    bag_of_features = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(extract_features_from_image)(path, step_size) 
        for path in image_paths
    )
    
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    print(f"Feature extraction took {time() - start_extract:.2f}s")
    print(f"Total descriptors: {len(bag_of_features)}")
    
    print("Compute vocab with K-means...")
    start_time = time()
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    end_time = time()
    print(f"K-means clustering took {end_time - start_time:.2f}s")
    
    return vocab