from django.shortcuts import render
from django.http import HttpResponseRedirect
#from django.core.urlresolvers import reverse
from django.urls import reverse
from covidapp.models import PicUpload
from covidapp.forms import ImageForm

# Create your views here.
def index(request):
    return render(request, 'index.html')


def list(request):
    image_path = ''
    #image_path1 = ''
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('list'))

    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        image_path = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request, 'list.html',
    {'documents':documents, 'image_path': image_path, 'form':form}
    )
##**********************COVID19 DETECTION WITH X-RAY***************
#************************IMPORT ESSENTIAL*********************

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf
#import Conv2D
import os
import numpy as np
import pandas as pd
#import GlobalAveragePooling2D
#import MaxPooling2D
import matplotlib.pyplot as plt

#***********************IMAGE preprocessing *****************

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def prepare_flat(img_224):
    base_model = load_model('static/covid19.model')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat

DATASET_DIR = "static/dataset"
os.listdir(DATASET_DIR)
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def covid19(request):
    print("-----------------------")
    normal_images = []
    for image_path in glob.glob(DATASET_DIR + '/normal/*'):
        normal_images.append(mpimg.imread(image_path))

        fig = plt.figure()
        fig.suptitle('normal')
        plt.imshow(normal_images[0], cmap='gray')


        covid_images = []
        for image_path in glob.glob(DATASET_DIR + '/covid/*'):
            covid_images.append(mpimg.imread(image_path))

            fig = plt.figure()
            fig.suptitle('covid-positive')
            plt.imshow(covid_images[0], cmap='gray')
            print(len(normal_images))
            print(len(covid_images))
            IMG_W = 150
            IMG_H = 150
            CHANNELS = 3

            INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
            NB_CLASSES = 2
            EPOCHS = 40
            BATCH_SIZE = 20
            model=tf.keras.models.Sequential([

                tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(INPUT_SHAPE)),
                tf.keras.layers.MaxPooling2D(2,2),

                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),


                tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32,activation='relu'),
                #tf.keras.layers.Dropout(0.25)
                tf.keras.layers.Dense(1,activation='sigmoid')


            ])
            from tensorflow.keras.optimizers import RMSprop

            model.compile(loss='binary_crossentropy',
                          optimizer=RMSprop(lr=0.001),
                          metrics=['accuracy'])
            print(model.summary())
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               validation_split=0.3)

            train_generator = train_datagen.flow_from_directory(
                DATASET_DIR,
                target_size=(IMG_H, IMG_W),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                subset='training')

            validation_generator = train_datagen.flow_from_directory(
                DATASET_DIR,
                target_size=(IMG_H, IMG_W),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle= False,
                subset='validation')

            history = model.fit_generator(
                train_generator,
                steps_per_epoch = train_generator.samples // BATCH_SIZE,
                validation_data = validation_generator,
                validation_steps = validation_generator.samples // BATCH_SIZE,
                epochs = EPOCHS)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            print("training_accuracy", history.history['accuracy'][-1])
            print("validation_accuracy", history.history['val_accuracy'][-1])

            label = validation_generator.classes
            pred= model.predict(validation_generator)
            predicted_class_indices=np.argmax(pred,axis=1)
            labels = (validation_generator.class_indices)
            labels2 = dict((v,k) for k,v in labels.items())
            predictions = [labels2[k] for k in predicted_class_indices]
            print(predicted_class_indices)
            print (labels)
            print (predictions)
            src= 'pic_upload/'
            import os
            for image_file_name in os.listdir(src):
                if image_file_name.endswith(".jpg") :
                    os.remove(src + image_file_name)

                    K.clear_session()


                    return render(
                        request,
                        'results.html',context={'predicted_class_indices':predicted_class_indices,'labels':labels,
                                                'predictions':predictions,'ns':ns}
                    )
