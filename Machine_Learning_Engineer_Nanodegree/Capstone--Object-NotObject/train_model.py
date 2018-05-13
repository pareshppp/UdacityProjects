# References:
# https://www.pyimagesearch.com


# set matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use('Agg')

# import required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

from cnn_models import LeNet



def construct_argument_parser():
    # construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', type=str, required=True, \
                    help='path to input dataset')
    ap.add_argument('-m', '--model', type=str, required=True, \
                    help='path to output model')
    ap.add_argument('-r', '--resize', type=int, default=28, \
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-a', '--architecture', type=str, default='LeNet', \
                    help='architecture for the convolutionsl neural network')
    ap.add_argument('-o', '--object', type=str, default='Pineapple', \
                    help='classification object')
    args = vars(ap.parse_args())
    return args




class ModelTraining:

    def __init__(self, epochs=25, init_lr=1e-3, batch_size=32, seed=123):
        # initialize variables
        self.EPOCHS = epochs    # number of epochs
        self.INIT_LR = init_lr  # initial learning rate
        self.BS = batch_size    # image batch size
        self.seed = seed        # random seed


    def get_imagePaths(self, dataset_path):
        # grab the image paths
        imagePaths = sorted(list(paths.list_images(dataset_path)))
        return imagePaths


    def preprocess_image(self, imagePath, resize_shape=(28, 28)):
        # load the image, pre-process it and store it in data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, resize_shape)
        image = img_to_array(image)
        return image


    def extract_label_from_imagePath(self, imagePath, clf_object):
        # extract class labels from the image path and add to labels
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == clf_object else 0
        return label


    def scale_pixel_intensity(self, data):
        # scale pixel intensities to range [0,1]
        data = np.array(data, dtype='float') / 255.0
        return data


    def one_hot_encode(self, ytrain, ytest, nclasses=2):
        # one-hot-encode labels
        ytrain = to_categorical(ytrain, num_classes=nclasses)
        ytest = to_categorical(ytest, num_classes=nclasses)
        return ytrain, ytest


    def construct_image_augmentor(self):
        # construct image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
                    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                    horizontal_flip=True, fill_mode='nearest')
        return aug


    def initialize_model(self, architecture='LeNet', width=28, height=28, depth=3, nclasses=2):
        # initialize the model
        # if architecture == 'LeNet':
        #     model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
        # else:
            # model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)

        model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
        
        optm = Adam(lr=self.INIT_LR, decay=self.INIT_LR/self.EPOCHS)
        
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
        
        # print(model.summary())
        return model


    def fit_model(self, model, augmentation, xtrain, xtest, ytrain, ytest):
        # train the network
        hist = model.fit_generator(augmentation.flow(xtrain, ytrain, batch_size=self.BS),\
                            validation_data=(xtest, ytest), \
                            steps_per_epoch=len(xtrain)//self.BS,\
                            epochs=self.EPOCHS, verbose=1)
        return hist, model


    def plot_loss_accuracy(self, hist, save_path, clf_object):
        # plot the training loss and accuracy
        plt.style.use('ggplot')
        plt.figure()
        N = self.EPOCHS
        H = hist
        plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
        plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
        plt.title('Training Loss and Accuracy on ' + clf_object + '/not' + clf_object)
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='lower left')
        plt.savefig(save_path)










if __name__ == '__main__':
    # initialize parameters
    EPOCHS = 20           # number of epochs
    INIT_LR = 1e-4        # initial learning rate
    BS = 32               # image batch size
    SEED = 123            # random seed

    data = []           # store image data
    labels = []         # store image labels
    model = None        # create model variable

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object'] 
    # print(args['object'])

    # create class object
    MT = ModelTraining(epochs=EPOCHS, init_lr=INIT_LR-3, batch_size=BS, seed=SEED)

    print('[INFO] loading images...')
    # print(args['dataset'])

    # grab the image paths and randomly shuffle them
    imagePaths = MT.get_imagePaths(dataset_path=args['dataset'])
    random.seed(SEED)
    random.shuffle(imagePaths)

    # loop over all imagePaths
    for imagePath in imagePaths:
        # load the image, pre-process it and store it in data list
        preprocessed_image = MT.preprocess_image(imagePath, \
                                    resize_shape=(args['resize'], args['resize']))
        data.append(preprocessed_image)

        # extract class labels from the image path and add to labels
        extracted_label = MT.extract_label_from_imagePath(imagePath, clf_object)
        labels.append(extracted_label)

    # scale pixel intensities to range [0,1]
    data = MT.scale_pixel_intensity(data)
    labels = np.array(labels)

    # split train (75%) and test data (25%)
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, \
                                        test_size=0.25, random_state=123)

    # one-hot-encode labels
    ytrain, ytest = MT.one_hot_encode(ytrain, ytest, nclasses=2)

    # construct image generator for data augmentation
    augmentation = MT.construct_image_augmentor()

    
    # initialize and compile the model
    print('[INFO] compiling model...')

    # model = MT.initialize_model(architecture=args['architecture'], \
    #                 width=args['resize'], height=args['resize'], depth=3, nclasses=2)

    # if args['architecture'] == 'LeNet':
    #     model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
    # else:
    #     model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)

    model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
    optm = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])

    # print model summary
    print(model.summary())

    # train the network
    print('[INFO] training model...')

    hist, model = MT.fit_model(model, augmentation, xtrain, xtest, ytrain, ytest)
    print('[DONE]')


    # save model to disk
    print('[INFO] saving model...')
    save_filename = args['model'] + '_' + args['architecture'] + '_' + str(args['resize']) + 'x' + str(args['resize']) + '.model'
    model.save(save_filename)

    # plot the training loss and accuracy
    print('[INFO] saving plot...')
    plot_save_filename = args['model'] + '_' + args['architecture'] + '_' + str(args['resize']) + 'x' + str(args['resize']) + '.png'
    MT.plot_loss_accuracy(hist, plot_save_filename, clf_object)
