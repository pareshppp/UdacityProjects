# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
import argparse
from imutils import paths
import random
import cv2
import os


def construct_argument_parser():
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, \
                    help='path to the trained model')
    ap.add_argument('-d', '--dataset', required=True, \
                    help='path to input dataset -- image folder')
    ap.add_argument('-r', '--resize', type=int, default=28, \
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-o', '--object', type=str, default='Pineapple', \
                    help='classification object')
    args = vars(ap.parse_args())
    return args



class ModelTesting:

    def __init__(self):
        pass


    def get_imagePaths(self, dataset_path):
        # grab the image paths
        imagePaths = sorted(list(paths.list_images(dataset_path)))
        return imagePaths


    def extract_label_from_imagePath(self, imagePath, clf_object):
        # extract class labels from the image path and add to labels
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == clf_object else 0
        return label


    def preprocess_image(self, imagePath, resize_shape=(28, 28)):
        # pre-process the image for classification
        image = cv2.imread(imagePath)
        image = cv2.resize(image, resize_shape) # resize
        image = image.astype('float') / 255.0   # normalize
        image = img_to_array(image)             # to array
        image = np.expand_dims(image, axis=0)   # add batch dim
        return image


    def classify_image(self, image, clf_object):
        # classify the input image
        (notObject, Object) = model.predict(image)[0]

        # get the prediction
        pred = 1 if Object > notObject else 0
        prob = Object if Object > notObject else -1 * notObject
        return pred, prob






if __name__ == '__main__':

    data = []
    predictions = []
    predict_proba = []
    true_labels = []
    accuracy_df = pd.DataFrame()

    SEED = 123

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object'] 

    # create class object
    MT = ModelTesting()

    # load the trained cnn model
    print('[INFO] loading model...')
    model = load_model(args['model'])

    # grab the image paths
    imagePaths = MT.get_imagePaths(dataset_path=args['dataset'])
    random.seed(SEED)
    random.shuffle(imagePaths)


    # loop over all imagePaths and get prediction
    print('[INFO] processing and classifying images...')
    for imagePath in imagePaths:
        # load the image, pre-process it and store the imagePath in data list
        preprocessed_image = MT.preprocess_image(imagePath, \
                                    resize_shape=(args['resize'], args['resize']))
        data.append(imagePath)
        
        # classify the input image and build the label
        pred, proba = MT.classify_image(preprocessed_image, clf_object)
        predictions.append(pred)
        predict_proba.append(proba)

        # extract class labels from the image path and add to labels
        extracted_label = MT.extract_label_from_imagePath(imagePath, clf_object)
        true_labels.append(extracted_label)
    
    print('[DONE]')
    

    # compare predictions with true labels
    accuracy_df['image_path'] = np.array(data)
    accuracy_df['predicted_probability'] = np.array(predict_proba)
    accuracy_df['predicted_labels'] = np.array(predictions)
    accuracy_df['true_labels'] = np.array(true_labels)
    accuracy_df['match'] = accuracy_df['true_labels'] == accuracy_df['predicted_labels']

    # write to file
    print('[INFO] saving prediction data to file...')
    save_filename = args['model'].split('.')[0] + '.csv'
    accuracy_df.to_csv(save_filename)

    # calculate accuracy
    accuracy = sum(accuracy_df['match']) / len(accuracy_df['match'])
    
    print('[OUTPUT] Testing Accuracy = {}'.format(accuracy))