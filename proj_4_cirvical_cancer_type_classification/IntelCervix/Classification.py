 # -*- coding:utf-8 -*-
__author__ = "Wang Hewen"
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(240, 320, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def prepare_data(batch_size):

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range=180,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            #'\\\\WANGHEWEN\\g\\Kaggle\\preprocessed\\train',  # this is the target directory
            'H:\\Kaggle\\preprocessed\\train',  # this is the target directory
            target_size=(240, 320),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            #'\\\\WANGHEWEN\\g\\Kaggle\\preprocessed\\additional',
            'H:\\Kaggle\\preprocessed\\additional',
            target_size=(240, 320),
            batch_size=batch_size,
            class_mode='categorical')
    return train_generator, validation_generator

def main():
    #preprocessed_folder = "G:/Kaggle/preprocessed"

    #datagen = ImageDataGenerator(
    #    rotation_range=180,
    #    width_shift_range=0.2,
    #    height_shift_range=0.2,
    #    shear_range=0.2,
    #    zoom_range=0.2,
    #    horizontal_flip=True,
    #    fill_mode='nearest')

    #img = load_img(os.path.join(preprocessed_folder, "train/Type_1/0.jpg"))  # this is a PIL image
    #x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    #x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    ## the .flow() command below generates batches of randomly transformed images
    ## and saves the results to the `preview/` directory
    #i = 0
    #for batch in datagen.flow(x, batch_size=1,
    #                          save_to_dir='./preview', save_prefix='cat', save_format='jpeg'):
    #    i += 1
    #    if i > 20:
    #        break  # otherwise the generator would loop indefinitely

    batch_size = 16
    model = build_model()
    train_generator, validation_generator = prepare_data(batch_size)

    #model.fit_generator(
    #    train_generator,
    #    steps_per_epoch=2000 // batch_size,
    #    epochs=500,
    #    validation_data=validation_generator,
    #    validation_steps=800 // batch_size,
    #    callbacks = [ModelCheckpoint("best_model.h5", verbose = 1, save_best_only = True, save_weights_only = True)])
    #model.save_weights('first_try.h5')  # always save your weights after training or during training


    model.load_weights("best_model.h5")
    print(model.predict_generator(validation_generator, 10, verbose=1))
    print(model.evaluate_generator(validation_generator, 100))

    return

if __name__ == "__main__":
    main()