import cv2 as cv
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import glob
from utils import read_img, resize_img


DATA_PATH = 'marvel_aug'
IMAGE_CATEGORIES = ['black widow', 'captain america', 'doctor strange', 'hulk', 'ironman',
    'loki', "spider-man", "thanos"]
SIFT_MAX_FEATURES = 50

#they were downsampled to deal with class imbalance, so we have the same amount 
#of images per class
n_cat = 50

#After our data augmentation and downsampling there are 1049 classes in train, 263 in valid, and


#which set is going to be used. Use val for validation and train for training set. Or test when testing the last model of svms. 
setused = "test"
#this set is automatically ignored when setting robustness=True

#name to recognise the experiment
experiment_parameters="svm_poly_c=2_deg=2"

#best model tried:
#"svm_poly_c=2_deg=2"
#use when having robustness=True

#the parameters used: 
c_value = 2
degree = 2
kernel = "poly"

#here we define a variable to recognize whether we are training and validating a model 
#or simply doing a robustness check, if True we must have specified the path to the correct folder
robustness = True

def get_image_paths(data_path, categories, num_per_cat, setused):
    '''
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    '''

    num_categories = len(categories) # number of scene categories.

    # This paths for each training and test image. By default it will have 1500
    # entries (15 categories * 100 training and test examples each)
    image_paths = [None] * (num_categories * num_per_cat)

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.
    labels = [None] * (num_categories * num_per_cat)


    for i,cat in enumerate(categories):
        images = glob.glob(os.path.join(data_path, setused, cat, '*.jpg'))
        print(images)
        for j in range(num_per_cat):
            image_paths[i * num_per_cat + j] = images[j]
            labels[i * num_per_cat + j] = cat

    return (image_paths, labels)


def build_codebook(image_paths, num_tokens=15):
    sift = cv.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    #sift = cv.SIFT_create()
    container = []
    for image_path in image_paths:
        img = resize_img(read_img(image_path, mono=True), size=(224,224))
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            container.append(descriptors)
    container = np.concatenate(container)
    print(container.shape)
    print('Training KMeans...')
    kmeans = KMeans(n_clusters=num_tokens)
    kmeans.fit(container)
    print('Done')
    return kmeans.cluster_centers_


def bag_of_words(image_paths, codebook):
    sift = cv.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    codebook_size = codebook.shape[0]
    image_features = []
    for image_path in image_paths:
        img = resize_img(read_img(image_path, mono=True), size=(224,224))
        keypoints, descriptors = sift.detectAndCompute(img, None)
        bow = np.zeros(codebook_size)
        if descriptors is not None:
            distances = cdist(descriptors, codebook)
            for d in distances:
                bow[np.argmin(d)] += 1
        image_features.append(bow.reshape(1, codebook_size))
    image_features = np.concatenate(image_features)
    return image_features


if not robustness:

    set_image_paths, labels =\
        get_image_paths(DATA_PATH, IMAGE_CATEGORIES, n_cat, setused)

    
    if setused =="train":
    
        codebook = build_codebook(set_image_paths)
        joblib.dump(codebook, "codebook_"+experiment_parameters+".joblib")
        
    else:
    
        codebook = joblib.load("codebook_"+experiment_parameters+".joblib")

    
    scaler = StandardScaler()

    print('Generating BOW features for set...')
    set_images_bow = bag_of_words(set_image_paths, codebook)
    
    if setused == "train":
        bows_scaled = scaler.fit_transform(set_images_bow)
        joblib.dump(scaler, "scaler"+experiment_parameters+".joblib")

    else: 
        scaler = joblib.load("scaler"+experiment_parameters+".joblib")
        bows_scaled = scaler.transform(set_images_bow)
        
    svm = SVC(gamma='scale', decision_function_shape="ovo")    

    if setused =="train":
        svm.fit(bows_scaled, labels)
        joblib.dump(svm, "svm"+experiment_parameters+".joblib")
    else:
        svm = joblib.load("svm"+experiment_parameters+".joblib")
    
    
    preds = svm.predict(bows_scaled)
    accuracy = f1_score(preds, labels, average="micro")#we choose macro since we avoided class imbalances by downsampling 


    print('Classification accuracy of SVM with BOW features in: '+setused+
    " with "+experiment_parameters, accuracy)

else:
    
    folders=["valid_with_decr_brightness","valid_with_incr_brightness","valid_with_decreasing_contrast","valid_with_increasing_contrast","valid_with_gaussblurr","valid_with_gaussnois","valid_with_occlusion","valid_with_s&p"]
    for folder in folders:
        
        abs_path_folder = os.path.abspath(folder)
        variations = os.listdir(abs_path_folder)
    
        amount_images_in_variation = 50 #the sets for performing robustness where constructed so that each class would always have 50
        variation_=[]
        f1scores=[]
        for variation in variations:
    
            path_variation = os.path.join(abs_path_folder, variation)
            print(path_variation)
        
            image_paths = [None] * (len(IMAGE_CATEGORIES) * amount_images_in_variation)
            labels = [None] * (len(IMAGE_CATEGORIES) * amount_images_in_variation)
        
            for i,cat in enumerate(IMAGE_CATEGORIES):
                images = glob.glob(os.path.join(path_variation, cat, '*.jpg'))
            
                for j in range(amount_images_in_variation):
                    image_paths[i * amount_images_in_variation + j] = images[j]
                    labels[i * amount_images_in_variation + j] = cat
        
            codebook = joblib.load("codebook_"+experiment_parameters+".joblib")
            set_images_bow = bag_of_words(image_paths, codebook)
            scaler = joblib.load("scaler"+experiment_parameters+".joblib")
            bows_scaled = scaler.transform(set_images_bow)
            svm = joblib.load("svm"+experiment_parameters+".joblib")
            preds = svm.predict(bows_scaled)
            f1score = f1_score(preds, labels, average="macro")#we choose macro since we avoided class imbalances by downsampling 
            variation_.append(variation)
            f1scores.append(f1score)

        print("-----------------------------------")
        print(folder)
        print("Variation factors",variation_)
        print("F1-scores", f1scores)
        print("--------------------------------------")
        
        
        
        
        
#Results obtained: 


#on original data svm_linear_c=1_deg=1_orginaldata has train f1 0.3262 and validation f1score 0.2175

#"svm_linear_c=1" has train F1-score 0.3469 and validation F1-score 0.24023   

#svm_poly_c=2_deg=2 has trainf1 0.344137 and validation f1 score  0.240019
        
#"svm_poly_c=2_deg=4" has trainf1 0.3450905 and validation  f1 0.240019     
        
#svm_poly_c=2_deg=8 has train f1 0.34580553 and validation f1 score 0.240722665    

#"svm_rbf_c=2_deg=4" has train f1 score 0.3454480 and validation f1 score 0.239068

#svm_poly_c=4_deg=10 has train f1 score 0.34449475 and validation f1 score 0.2381178