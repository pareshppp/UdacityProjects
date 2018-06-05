# set matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use('Agg')

# import required packages
import argparse
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imutils import paths

from models import LeNet
from models import FullConn
from models import CustomConv



def construct_argument_parser():
    # construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', type=str, required=True, \
                    help='path to input/Training dataset')
    ap.add_argument('-m', '--model', type=str, required=True, \
                    help='name of output model')
    ap.add_argument('-r', '--resize', type=int, default=28, \
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-a', '--architecture', type=str, default='LeNet', \
                    help='architecture for the convolutional neural network')
    ap.add_argument('-o', '--object', type=str, default='Pineapple', \
                    help='classification object')
    args = vars(ap.parse_args())
    return args




class ModelTraining:

    def __init__(self, epochs=25, lr=1e-3, batch_size=32, seed=123):
        # initialize variables
        self.epochs = epochs            # number of epochs
        self.lr = lr                    # initial learning rate
        self.batch_size = batch_size    # image batch size
        self.seed = seed                # random seed


    def get_image_paths(self, dataset_path):
        # grab the image paths
        image_paths = sorted(list(paths.list_images(dataset_path)))
        return image_paths


    def preprocess_image(self, image_path, resize_shape=(28, 28)):
        # load the image, pre-process it and store it in data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, resize_shape)
        image = img_to_array(image)
        return image


    def extract_label_from_image_path(self, image_path, clf_object):
        # extract class labels from the image path and add to labels
        label = image_path.split(os.path.sep)[-2]
        label = 1 if label == clf_object else 0
        return label


    def scale_pixel_intensity(self, data):
        # scale pixel intensities to range [0,1]
        data = np.array(data, dtype='float') / 255.0
        return data


    def one_hot_encode(self, y_train, y_test, nclasses=2):
        # one-hot-encode labels
        y_train = to_categorical(y_train, num_classes=nclasses)
        y_test = to_categorical(y_test, num_classes=nclasses)
        return y_train, y_test


    def construct_image_augmentor(self):
        # construct image generator for data augmentation
        augment = ImageDataGenerator(
                    rotation_range=270, 
                    width_shift_range=0.5, 
                    height_shift_range=0.5, 
                    shear_range=0.5, 
                    zoom_range=0.5,
                    channel_shift_range=0.5,
                    horizontal_flip=True,
                    vertical_flip=True, 
                    fill_mode='reflect')
        return augment


    # def initialize_model(self, architecture='LeNet', width=28, height=28, depth=3, nclasses=2):
    #     # initialize the model
    #     # if architecture == 'LeNet':
    #     #     model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
    #     # else:
    #         # model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)

    #     model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
        
    #     optm = Adam(lr=self.lr, decay=self.lr/self.epochs)
        
    #     model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
        
    #     # print(model.summary())
    #     return model


    def fit_model(self, model, augmentation, x_train, x_test, y_train, y_test):
        # train the network
        hist = model.fit_generator(
                            augmentation.flow(x_train, y_train, batch_size=self.batch_size),
                            validation_data=(x_test, y_test),
                            steps_per_epoch=len(x_train)//self.batch_size,
                            epochs=self.epochs, 
                            verbose=1)
        return hist, model


    def plot_loss_accuracy(self, hist, save_path, clf_object):
        # plot the training loss and accuracy
        plt.style.use('ggplot')
        plt.figure()
        N = self.epochs
        H = hist
        plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
        plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
        plt.title('Training Loss and Accuracy on ' + clf_object + '/not' + clf_object)
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.ylim(0, 1)
        plt.legend(loc='lower left')
        plt.savefig(save_path)










if __name__ == '__main__':
    # initialize parameters
    epochs = 35             # number of epochs
    lr = 1e-4               # initial learning rate
    batch_size = 32         # image batch size
    seed = 123              # random seed

    image_data = []         # store image data
    image_labels = []       # store image labels
    model = None            # create model variable

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object'] 
    # print(args['object'])

    # create class object
    MT = ModelTraining(epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)

    print('[INFO] loading images...')
    # print(args['dataset'])

    # grab the image paths and randomly shuffle them
    image_paths = MT.get_image_paths(dataset_path=args['dataset'])
    random.seed(seed)
    random.shuffle(image_paths)

    # loop over all image_paths
    for image_path in image_paths:
        # load the image, pre-process it and store it in data list
        preprocessed_image = MT.preprocess_image(image_path, \
                                    resize_shape=(args['resize'], args['resize']))
        image_data.append(preprocessed_image)

        # extract class labels from the image path and add to labels
        extracted_label = MT.extract_label_from_image_path(image_path, clf_object)
        image_labels.append(extracted_label)

    # scale pixel intensities to range [0,1]
    image_data = MT.scale_pixel_intensity(image_data)
    image_labels = np.array(image_labels)

    # split train (75%) and test data (25%)
    x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, \
                                        test_size=0.25, random_state=123)

    # one-hot-encode labels
    y_train, y_test = MT.one_hot_encode(y_train, y_test, nclasses=2)

    # construct image generator for data augmentation
    augmentation = MT.construct_image_augmentor()

    
    # initialize and compile the model
    print('[INFO] initializing model...')

    # model = MT.initialize_model(architecture=args['architecture'], \
    #                 width=args['resize'], height=args['resize'], depth=3, nclasses=2)

    if args['architecture'] == 'LeNet':
        print('[INFO] using LeNet architecture...')
        model = LeNet.build_model(width=args['resize'], height=args['resize'], \
                                  depth=3, nclasses=2)
        # save model visualization
        plot_model(model, to_file='Output_Files/Save_Plot/LeNet.png', \
                   show_layer_names=True, show_shapes=True)

    elif args['architecture'] == 'FullConn':
        print('[INFO] using FullConn architecture...')
        model = FullConn.build_model(width=args['resize'], height=args['resize'], \
                                     depth=3, nclasses=2)
        # save model visualization
        plot_model(model, to_file='Output_Files/Save_Plot/FullConn.png', \
                   show_layer_names=True, show_shapes=True)
        
    elif args['architecture'] == 'CustomConv':
        print('[INFO] using CustomConv architecture...')
        model = CustomConv.build_model(width=args['resize'], height=args['resize'], \
                                       depth=3, nclasses=2)
        # save model visualization
        plot_model(model, to_file='Output_Files/Save_Plot/CustomConv.png', \
                   show_layer_names=True, show_shapes=True)

    else:
        print('[INFO] using LeNet architecture...')
        model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
        # save model visualization
        plot_model(model, to_file='Output_Files/Save_Plot/LeNet.png', \
                   show_layer_names=True, show_shapes=True)


    # model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)


    print('[INFO] initializing optimizer...')
    optm = Adam(lr=lr, decay=lr/epochs)

    print('[INFO] compiling model...')
    model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])


    print('[INFO] model summary...')
    print(model.summary())


    # train the network
    print('[INFO] training model...')

    hist, model = MT.fit_model(model, augmentation, x_train, x_test, y_train, y_test)
    print('[DONE]')


    # save model to disk
    print('[INFO] saving model...')
    save_filename = args['model'] + '_' + args['architecture'] + \
                                    '_' + str(args['resize']) + \
                                    'x' + str(args['resize'])

    save_filename = 'Output_Files/Save_Model/' + save_filename + '.model'

    model.save(save_filename)

    # plot the training loss and accuracy
    print('[INFO] saving plot...')
    plot_save_filename = args['model'] + '_' + args['architecture'] + \
                                         '_' + str(args['resize']) + \
                                         'x' + str(args['resize'])

    plot_save_filename = 'Output_Files/Save_Plot/' + plot_save_filename + '.png'

    MT.plot_loss_accuracy(hist, plot_save_filename, clf_object)

    
