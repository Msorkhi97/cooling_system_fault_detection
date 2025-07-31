# Third-Party Library Imports
import tensorflow.keras as keras

# Local Application Imports
from src.model import my_model


class TrainModel():
    def __init__(self,config,x_train,y_train):
        self.config = config
        self.x_train = x_train
        self.y_train = y_train

    def train(self):


        model = my_model((self.config["data"]["configuration"]["window_size"], self.config["model"]["configuration"]["num_feature"]))

        callbacks = [
            keras.callbacks.ModelCheckpoint(self.config["output"]["path"] + "/best_model.keras", save_best_only=True, monitor="val_loss"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=300000, verbose=1)
        ]

        model.compile(
            loss=self.config["training"]["loss"],  # Use binary cross-entropy for two classes
            optimizer=self.config["training"]["optimizer"],
            metrics=self.config["training"]["metrics"]
        )

        model.fit(
            self.x_train,
            self.y_train,
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
            callbacks=callbacks,
            validation_split=self.config["training"]["val_size"],
            verbose=1,
        )

