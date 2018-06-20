# import necessary packages

import argparse
import random
import os
import re
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths




def construct_argument_parser():
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True,
                    help='path to the trained model')
    ap.add_argument('-d', '--dataset', required=True,
                    help='path to input/Testing dataset')
    ap.add_argument('-r', '--resize', type=int, default=28,
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-o', '--object', type=str, default='Pineapple',
                    help='classification object')
    args = vars(ap.parse_args())
    return args


class ModelTesting:

    def __init__(self):
        pass

    def get_image_paths(self, dataset_path):
        # grab the image paths
        image_paths = sorted(list(paths.list_images(dataset_path)))
        return image_paths

    def extract_label_from_image_path(self, image_path, clf_object):
        # extract class labels from the image path and add to labels
        label = image_path.split(os.path.sep)[-2]
        label = 1 if label == clf_object else 0
        return label

    def preprocess_image(self, image_path, resize_shape=(28, 28)):
        # pre-process the image for classification
        image = cv2.imread(image_path)
        image = cv2.resize(image, resize_shape)  # resize
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

    image_data = []                 # store image data
    predictions = []                # store model predictions
    predict_proba = []              # store model prediction probabilities
    true_labels = []                # store true image labels
    accuracy_df = pd.DataFrame()    # store model accuracy data
    prediction_df = pd.DataFrame()  # store model prediction data

    seed = 123

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object']

    # create class object
    MT = ModelTesting()

    # load the trained cnn model
    print('\n[INFO] loading model...\n')
    model = load_model(args['model'])

    # grab the image paths
    image_paths = MT.get_image_paths(dataset_path=args['dataset'])
    random.seed(seed)
    random.shuffle(image_paths)

    # loop over all image_paths and get prediction
    print('\n[INFO] processing and classifying images...\n')
    for image_path in image_paths:
        # load the image, pre-process it and store the image_path in data list
        preprocessed_image = MT.preprocess_image(image_path,
                                                 resize_shape=(args['resize'], args['resize']))
        # image_data.append(image_path)

        # classify the input image and build the label
        pred, proba = MT.classify_image(preprocessed_image, clf_object)
        predictions.append(pred)
        predict_proba.append(proba)

        # extract class labels from the image path and add to labels
        extracted_label = MT.extract_label_from_image_path(image_path, clf_object)
        true_labels.append(extracted_label)

    print('\n[DONE]\n')

    # compare predictions with true labels
    prediction_df['image_path'] = np.array(image_paths)
    prediction_df['predicted_probability'] = np.array(predict_proba)
    prediction_df['predicted_labels'] = np.array(predictions)
    prediction_df['true_labels'] = np.array(true_labels)
    prediction_df['label_match'] = prediction_df['true_labels'] == prediction_df['predicted_labels']


    # write predictions to file
    model_name = re.split('\/|\.', args['model'])[-2]
    pred_filename = 'Output_Files/Save_Output_Accuracy/' + model_name + '_predictions.csv'

    print('\n[INFO] saving prediction data to file ', pred_filename, '\n')
    prediction_df.to_csv(pred_filename)

    # calculate accuracy
    accuracy = sum(prediction_df['label_match']) / len(prediction_df['label_match'])
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('[OUTPUT] Testing Accuracy = {}'.format(accuracy))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++\n')

    # store accuracy data
    accuracy_df['model_name'] = pd.Series(model_name)
    accuracy_df['testing_accuracy'] = pd.Series(accuracy)

    # write accuracy data to file
    acc_filename = 'Output_Files/Save_Output_Accuracy/Pineapple-NotPineapple_accuracy.csv'
    print('\n[INFO] saving accuracy data to file ', acc_filename, '\n')

    # if file exists then append, else write
    if os.path.exists(acc_filename):
        with open(acc_filename, 'a') as f:
            accuracy_df.to_csv(f, header=False)
    else:
        with open(acc_filename, 'w+') as f:
            accuracy_df.to_csv(f)



    
