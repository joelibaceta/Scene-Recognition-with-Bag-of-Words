"""
Scene Recognition Comparison: SIFT vs SURF vs BRIEF
Compares the performance of bag-of-words with SVM classifier using three different features.
"""
from __future__ import print_function
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from get_image_paths import get_image_paths
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from build_vocabulary_surf import build_vocabulary_surf
from get_bags_of_surf import get_bags_of_surf
from build_vocabulary_brief import build_vocabulary_brief
from get_bags_of_brief import get_bags_of_brief
from svm_classify import svm_classify

# Configuration
DATA_PATH = '../data/'
CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
ABBR_CATEGORIES = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']
NUM_TRAIN_PER_CAT = 100
VOCAB_SIZE = 400

def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories, title):
    """Build and plot confusion matrix"""
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm_normalized, abbr_categories, title=title)

def evaluate_feature(feature_name, train_image_paths, test_image_paths, 
                     train_labels, test_labels):
    """
    Evaluate a specific feature extractor with SVM classifier.
    
    Args:
        feature_name: 'sift', 'surf', or 'brief'
        train_image_paths: paths to training images
        test_image_paths: paths to test images
        train_labels: training labels
        test_labels: test labels
        
    Returns:
        accuracy: overall accuracy
        predicted_categories: predicted labels
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {feature_name.upper()} features with SVM classifier")
    print(f"{'='*60}")
    
    # Build vocabulary
    if feature_name == 'sift':
        vocab_file = 'vocab_sift.pkl'
        if not os.path.isfile(vocab_file):
            print('Building SIFT vocabulary...')
            vocab = build_vocabulary(train_image_paths, VOCAB_SIZE)
            with open(vocab_file, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Extract features
        if not os.path.isfile('train_feats_sift.pkl'):
            train_image_feats = get_bags_of_sifts(train_image_paths)
            with open('train_feats_sift.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_feats_sift.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)
        
        if not os.path.isfile('test_feats_sift.pkl'):
            test_image_feats = get_bags_of_sifts(test_image_paths)
            with open('test_feats_sift.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_feats_sift.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    
    elif feature_name == 'surf':
        vocab_file = 'vocab_surf.pkl'
        if not os.path.isfile(vocab_file):
            print('Building SURF vocabulary...')
            vocab = build_vocabulary_surf(train_image_paths, VOCAB_SIZE)
            with open(vocab_file, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Extract features
        if not os.path.isfile('train_feats_surf.pkl'):
            train_image_feats = get_bags_of_surf(train_image_paths, vocab_file)
            with open('train_feats_surf.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_feats_surf.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)
        
        if not os.path.isfile('test_feats_surf.pkl'):
            test_image_feats = get_bags_of_surf(test_image_paths, vocab_file)
            with open('test_feats_surf.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_feats_surf.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    
    elif feature_name == 'brief':
        vocab_file = 'vocab_brief.pkl'
        if not os.path.isfile(vocab_file):
            print('Building BRIEF vocabulary...')
            vocab = build_vocabulary_brief(train_image_paths, VOCAB_SIZE)
            with open(vocab_file, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Extract features
        if not os.path.isfile('train_feats_brief.pkl'):
            train_image_feats = get_bags_of_brief(train_image_paths, vocab_file)
            with open('train_feats_brief.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_feats_brief.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)
        
        if not os.path.isfile('test_feats_brief.pkl'):
            test_image_feats = get_bags_of_brief(test_image_paths, vocab_file)
            with open('test_feats_brief.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_feats_brief.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    
    # Classify using SVM
    print('Classifying with SVM...')
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
    
    # Calculate accuracy
    accuracy = float(len([x for x in zip(test_labels, predicted_categories) 
                         if x[0] == x[1]])) / float(len(test_labels))
    
    print(f"\n{feature_name.upper()} Overall Accuracy = {accuracy:.4f}")
    
    # Per-category accuracy
    print("\nPer-category accuracy:")
    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels, predicted_categories) 
                                  if x[0] == x[1] and x[0] == category])) / float(test_labels.count(category))
        print(f"  {category}: {accuracy_each:.4f}")
    
    return accuracy, predicted_categories

def main():
    """Main comparison function"""
    print("Loading image paths...")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    
    # Store results
    results = {}
    
    # Evaluate each feature type
    for feature_name in ['sift', 'surf', 'brief']:
        accuracy, predicted_categories = evaluate_feature(
            feature_name, train_image_paths, test_image_paths, 
            train_labels, test_labels
        )
        results[feature_name] = {
            'accuracy': accuracy,
            'predictions': predicted_categories
        }
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for feature_name, result in results.items():
        print(f"{feature_name.upper()}: {result['accuracy']:.4f}")
    
    # Plot confusion matrices
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    
    fig = plt.figure(figsize=(18, 6))
    
    for idx, (feature_name, result) in enumerate(results.items(), 1):
        predicted_categories_ids = [CATE2ID[x] for x in result['predictions']]
        
        plt.subplot(1, 3, idx)
        cm = confusion_matrix(test_labels_ids, predicted_categories_ids)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(
            cm_normalized, ABBR_CATEGORIES, 
            title=f'{feature_name.upper()} (Acc: {result["accuracy"]:.4f})'
        )
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'comparison_results.png'")
    plt.show()
    
    # Bar chart comparison
    plt.figure(figsize=(10, 6))
    features = list(results.keys())
    accuracies = [results[f]['accuracy'] for f in features]
    
    bars = plt.bar([f.upper() for f in features], accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Feature Type', fontsize=12)
    plt.title('Scene Recognition: Feature Comparison (SVM Classifier)', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("Accuracy comparison plot saved as 'accuracy_comparison.png'")
    plt.show()

if __name__ == '__main__':
    main()
