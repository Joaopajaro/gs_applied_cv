import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

tf.random.set_seed(42)
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

datagen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
datagen.fit(x_train)

model = keras.Sequential([
    # Bloco 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Bloco 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Bloco 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Classificador
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.summary()

model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("model.keras", monitor="val_accuracy", save_best_only=True),
]

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {acc * 100:.2f}%")

labels = ["aviao", "carro", "passaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhao"]
np.save("labels.npy", labels)
print("Modelo salvo em model.keras | Labels em labels.npy")
