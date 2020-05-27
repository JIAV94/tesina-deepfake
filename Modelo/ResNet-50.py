# transfer_learn.py
# This program is an example of usinf Transfer Learning. Transfer learning let apply the power of a n existing powerful
# trained model to a dataset we are interested in. In this example, we will use the Inception-V3 model
# This code was inspired by the post https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Supress warning and informational messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import glob
import matplotlib.pyplot as plt

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# Get count of number of files in this folder and all subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Get count of number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

# Define image generators that will variations of image with the image rotated slightly, shifted up, down, left, or right
# sheared, zoomed in, or flipped horizontally on the verical axis (ie. person looking to the left ends up looking to the right)


# Main Code
Image_width, Image_height = 150, 150
Training_Epochs = 5
Batch_Size = 128
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

num_epoch = Training_Epochs
batch_size = Batch_Size

# Define data pre-processing
# Define image generators for training and testing
train_image_gen = ImageDataGenerator( rescale=1./255., preprocessing_function=preprocess_input)
test_image_gen = ImageDataGenerator( rescale=1./255., preprocessing_function=preprocess_input)

# Connect the image generator to a folder contains the source images the image generator alters.
# Training image generator
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed = 42   # set seed for reproducability
)

# Validation image generator
validation_generator = test_image_gen.flow_from_directory(
    validate_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed = 42   # set seed for reproducability
)

# Load the INception V3 model and load it with it's pre-trained weights. But exclude the final
# Fully Connected Layer
ResNet50_base_model = ResNet50(weights='imagenet', include_top=False) # include_top=False excludes final FC layer
print('ResNet50 base model without last FC loaded')

# Define the layers in the new classification prediction
x = ResNet50_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)  # new FC layer, random init
predictions = Dense(num_classes, activation='softmax')(x)   # new softmax layer

# Define trainable model which links input from the ResNet50 base model to the new classification prediction layers
model = Model(inputs=ResNet50_base_model.input, outputs=predictions)

# Print model structure diagram
print(model.summary())

# Option 1: Basic Transfer Learning
print ('\nPerforming Transfer Learning')

# Freeze all layers in the ResNet50 base model
for layer in ResNet50_base_model.layers:
    layer.trainable = False

# Define model compile for basic Transfer Learning
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the transfer learning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images from
# the training or validation folders and alter them slightly
history_transfer_learning = model.fit_generator(
    train_generator,
    epochs = num_epoch,
    steps_per_epoch = num_train_samples // batch_size,
    validation_data = validation_generator,
    validation_steps = num_validate_samples // batch_size,
    class_weight = 'auto')

# Save transfer learning model
model.save('resnet50-transfer-learning.model')

# Option2: Transfer Learning with Fine-Tuning - retrain the end few layers (called the top layers) of the resnet50 model
print('\nFine tunning existing model')
# Freeze
Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
    layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Fine-tuning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images from the training or
# folders, alter then slightly, and pass the images back
history_fine_tune = model.fit_generator(
    train_generator,
    steps_per_epoch = num_train_samples // batch_size,
    epochs = num_epoch,
    validation_data = validation_generator,
    validation_steps = num_validate_samples // batch_size,
    class_weight = 'auto'
)

# Save fine tuned model
model.save('resnet50-fine-tune.model')
