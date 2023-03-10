import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import cv2

main_path = './archive/'
classes = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua',
           'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
           'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian',
           'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
           'wheaten_terrier', 'yorkshire_terrier']

# classes = ['american_bulldog', 'chihuahua','pomeranian',
#            'pug', 'samoyed', 'staffordshire_bull_terrier',
#            ]

img_size = (128, 128)
batch_size = 32

from keras.utils import image_dataset_from_directory

ulaz_trening = image_dataset_from_directory(main_path,
                                            subset='training',
                                            validation_split=0.3,
                                            image_size=img_size,
                                            batch_size=batch_size,
                                            seed=123)


print(ulaz_trening)
class_names=ulaz_trening.class_names

ulaz_val = image_dataset_from_directory(main_path,
                                        subset='validation',
                                        validation_split=0.3,
                                        image_size=img_size,
                                        batch_size=batch_size,
                                        seed=123)
val_batches = tf.data.experimental.cardinality(ulaz_val)
ulaz_test =ulaz_val.take((val_batches) // 3)
ulaz_val = ulaz_val.skip((val_batches) // 3)

plt.figure(figsize=(10,10))
for img, lab in ulaz_trening.take(1):
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(class_names[lab[i]])
        plt.axis('off')
plt.show()

from keras import Sequential
from keras import layers
from keras.losses import SparseCategoricalCrossentropy

data_augmentation = Sequential([
    layers.RandomFlip('horizontal', input_shape=(128, 128, 3)),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.3),
    layers.RandomTranslation(width_factor=0.1, height_factor=0.1),
    layers.RandomContrast(0.2, seed=5),
    layers.RandomBrightness(factor=0.2)
])
#

plt.figure(figsize=(10, 10))
for img, lab in ulaz_trening.take(1):
    for i in range(6):
        img_aug = data_augmentation(img)
        plt.subplot(1,6,i+1)
        plt.imshow(img_aug[0].numpy().astype('uint8'))
        plt.title(class_names[lab[i]])
        plt.axis('off')
plt.show()
from keras.regularizers import l2
def create_model(dropout_rate):
    model = Sequential([
        data_augmentation,

        layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Dropout(dropout_rate),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Dropout(dropout_rate),

        layers.Flatten(),

        layers.Dense(1024, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(len(classes), 'softmax')
    ])

    model.summary()


    model.compile('adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics='accuracy'
        )

    return model

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=6)
for i in range(1, 2):

    model = create_model(i/10)
    history = model.fit(ulaz_trening,
                        validation_data=ulaz_val,
                        epochs=100,
                        callbacks=es
                        )

    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['loss'], label ='train set' )
    plt.plot(history.history['val_loss'], label ='validation set' )
    plt.xlabel('Number of epochs')
    plt.title('Loss function')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['accuracy'], label = 'train set')
    plt.plot(history.history['val_accuracy'], label = 'validation set')
    plt.xlabel('Number of epochs')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    pred = np.array([])
    labels = np.array([])
    for img, lab in ulaz_test:
        labels = np.append(labels, lab)
        pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

    for img, lab in ulaz_trening:
        labels_train = np.append(labels, lab)
        pred_train = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))



    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    print('Accuracy on test set: ', accuracy_score(labels, pred))

    plt.figure()
    cm = confusion_matrix(labels, pred, normalize='true')
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmDisplay.plot(ax=ax, xticks_rotation='vertical')
    plt.title('Test set')
    plt.show()

    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(labels_train, pred_train, normalize='true')
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmDisplay.plot( ax=ax, xticks_rotation='vertical')
    plt.title('Train set')
    plt.show()

    plt.figure(figsize=(10, 10))
    for img, lab in ulaz_test.take(1):  # take(1) uzima jednu sliku, kada pozovemo ponovo dace nam neku drugu
        # plt.title(lab[i])
        pred1 = np.argmax(model.predict(img,verbose=0),axis=1)
        pred = np.append(pred, pred1)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.axis('off')
            if lab[i] != pred1[i]:
                plt.title('True class: '+ str(class_names[lab[i]])+ '\n predicted class: '+str(class_names[pred1[i]]))
            else:
                plt.title('True and predicted class:\n '+ str(class_names[lab[i]]))

    plt.show()



