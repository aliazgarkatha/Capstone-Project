import pandas as pd
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ReduceLROnPlateau

# Reading Annotation and image folders
train_images_folder = r"E:\AIDA\Semester 2\Capstone project\Data sets\Final dataset\FINALWildlife\FINALWildlife\train"
test_images_folder = r"E:\AIDA\Semester 2\Capstone project\Data sets\Final dataset\FINALWildlife\FINALWildlife\test"
train_annotations = pd.read_csv(r"E:\AIDA\Semester 2\Capstone project\Data sets\Final dataset\FINALWildlife\FINALWildlife\train_annotations.csv")
test_annotations = pd.read_csv(r"E:\AIDA\Semester 2\Capstone project\Data sets\Final dataset\FINALWildlife\FINALWildlife\test_annotations.csv")


# Handling missing values and duplicates
train_annotations.dropna(inplace=True)
test_annotations.dropna(inplace=True)
train_annotations = train_annotations.drop_duplicates(subset=['filename'])
test_annotations = test_annotations.drop_duplicates(subset=['filename'])

# Class column formatting on annotation files
train_annotations['class'] = train_annotations['class'].str.lower()
test_annotations['class'] = test_annotations['class'].str.lower()

# Resizing of images
target_size = (224, 224)
batch_size = 32

# Normalization and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# creating data generators for train and test dataset
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_annotations,
    directory=train_images_folder,
    x_col="filename",
    y_col="class",
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_annotations,
    directory=test_images_folder,
    x_col="filename",
    y_col="class",
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# Define the ResNet model
def identity_block(input_tensor, filters, kernel_size):
    filters1, filters2, filters3 = filters
    updated_tensor = Conv2D(filters1, (1, 1))(input_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)
    updated_tensor = Activation('relu')(updated_tensor)

    updated_tensor = Conv2D(filters2, kernel_size, padding='same')(updated_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)
    updated_tensor = Activation('relu')(updated_tensor)

    updated_tensor = Conv2D(filters3, (1, 1))(updated_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)

    updated_tensor = updated_tensor + input_tensor
    updated_tensor = Activation('relu')(updated_tensor)
    return updated_tensor

def convolutional_block(input_tensor, filters, kernel_size, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    updated_tensor = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)
    updated_tensor = Activation('relu')(updated_tensor)

    updated_tensor = Conv2D(filters2, kernel_size, padding='same')(updated_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)
    updated_tensor = Activation('relu')(updated_tensor)

    updated_tensor = Conv2D(filters3, (1, 1))(updated_tensor)
    updated_tensor = BatchNormalization()(updated_tensor)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    updated_tensor = updated_tensor + shortcut
    updated_tensor = Activation('relu')(updated_tensor)
    return updated_tensor

input_tensor = Input(shape=(224, 224, 3))
Updated_tensor = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
Updated_tensor = BatchNormalization()(Updated_tensor)
Updated_tensor = Activation('relu')(Updated_tensor)
Updated_tensor = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Updated_tensor)

# Increasing depth of layer stacking and adjusting the filters
Updated_tensor = convolutional_block(Updated_tensor, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1))
Updated_tensor = identity_block(Updated_tensor, filters=[64, 64, 256], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[64, 64, 256], kernel_size=(3, 3))

Updated_tensor = convolutional_block(Updated_tensor, filters=[128, 128, 512], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[128, 128, 512], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[128, 128, 512], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[128, 128, 512], kernel_size=(3, 3))

# Adding more layers
Updated_tensor = convolutional_block(Updated_tensor, filters=[256, 256, 1024], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[256, 256, 1024], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[256, 256, 1024], kernel_size=(3, 3))
Updated_tensor = identity_block(Updated_tensor, filters=[256, 256, 1024], kernel_size=(3, 3))

Updated_tensor = GlobalAveragePooling2D()(Updated_tensor)
output_tensor = Dense(len(train_generator.class_indices), activation='softmax')(Updated_tensor)

model = Model(input_tensor, output_tensor)


# Adding callback to monitor val_loss
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with callbacks
model.fit(train_generator, epochs=25, validation_data=test_generator, callbacks=[reduce_lr])


# Model evaluation
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the trained model
model.save("animal_detection_model.keras")


# Save class indices to a file
class_indices = train_generator.class_indices
with open("animal_detection_class_indices.txt", "w") as f:
    for class_name, class_index in class_indices.items():
        f.write(f"{class_name}:{class_index}\n")



