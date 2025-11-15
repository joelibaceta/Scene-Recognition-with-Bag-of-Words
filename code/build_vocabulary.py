from PIL import Image
import numpy as np
# Usar OpenCV en lugar de cyvlfeat
from cv2_sift_utils import dsift, kmeans
from time import time
from joblib import Parallel, delayed
import multiprocessing

import pdb

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    ##################################################################################
    # TODO:                                                                          #
    # Load images from the training set. To save computation time, you don't         #
    # necessarily need to sample from all images, although it would be better        #
    # to do so. You can randomly sample the descriptors from each image to save      #
    # memory and speed up the clustering. Or you can simply call vl_dsift with       #
    # a large step size here.                                                        #
    #                                                                                #
    # For each loaded image, get some SIFT features. You don't have to get as        #
    # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
    # trying to get a representative sample here.                                    #
    #                                                                                #
    # Once you have tens of thousands of SIFT features from many training            #
    # images, cluster them with kmeans. The resulting centroids are now your         #
    # visual word vocabulary.                                                        #
    ##################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # This function will sample SIFT descriptors from the training images,           #
    # cluster them with kmeans, and then return the cluster centers.                 #
    #                                                                                #
    # Function : dsift()                                                             #
    # SIFT_features is a N x 128 matrix of SIFT features                             #
    # There are step, bin size, and smoothing parameters you can                     #
    # manipulate for dsift(). We recommend debugging with the 'fast'                 #
    # parameter. This approximate version of SIFT is about 20 times faster to        #
    # compute. Also, be sure not to use the default value of step size. It will      #
    # be very slow and you'll see relatively little performance gain from            #
    # extremely dense sampling. You are welcome to use your own SIFT feature.        #
    #                                                                                #
    # Function : kmeans(X, K)                                                        #
    # X is a M x d matrix of sampled SIFT features, where M is the number of         #
    # features sampled. M should be pretty large!                                    #
    # K is the number of clusters desired (vocab_size)                               #
    # centers is a d x K matrix of cluster centroids.                                #
    #                                                                                #
    # NOTE:                                                                          #
    #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
    #        2. kmeans( ? , vocab_size)                                              #  
    #                                                                                #
    # ################################################################################
    '''
    Input : 
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    '''

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
        img = np.asarray(Image.open(path), dtype='float32')
        frames, descriptors = dsift(img, step=[step_size, step_size], fast=True)
        return descriptors
    
    bag_of_features = []
    
    print("Extract SIFT features (parallelized)")
    
    # Optimización: usar solo una muestra de imágenes y paralelizar
    sample_rate = 10  # Usar 1 de cada 10 imágenes
    step_size = 15    # Más grande = menos features = más rápido
    
    # Seleccionar imágenes a procesar
    sampled_paths = [path for i, path in enumerate(image_paths) if i % sample_rate == 0]
    print(f"Processing {len(sampled_paths)} images in parallel...")
    
    # Procesar en paralelo usando todos los cores disponibles
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} CPU cores")
    
    start_extract = time()
    bag_of_features = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(extract_features_from_image)(path, step_size) 
        for path in sampled_paths
    )
    
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    print(f"Feature extraction took {time() - start_extract:.2f}s")
    print(f"Total descriptors: {len(bag_of_features)}")
    
    print("Compute vocab with K-means...")
    start_time = time()
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    end_time = time()
    print("K-means clustering took ", (end_time - start_time), " seconds.")
    
    
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab