from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
# Usar OpenCV en lugar de cyvlfeat
from cv2_sift_utils import dsift
from time import time
from joblib import Parallel, delayed
import multiprocessing
import pdb

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    
    # Función auxiliar para procesar una imagen
    def process_image(path, vocab):
        img = np.asarray(Image.open(path), dtype='float32')
        frames, descriptors = dsift(img, step=[1,1], fast=True)
        dist = distance.cdist(vocab, descriptors, metric='euclidean')
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        return hist_norm
    
    start_time = time()
    print("Construct bags of sifts (parallelized)...")
    
    # Procesar en paralelo
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} CPU cores")
    
    image_feats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_image)(path, vocab) 
        for path in image_paths
    )
    
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print(f"Bag of SIFT construction took {end_time - start_time:.2f}s")
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    
    return image_feats