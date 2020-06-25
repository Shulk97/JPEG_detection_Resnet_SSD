from os.path import join

from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist

from abc import ABC, abstractmethod

class TemplateConfiguration(ABC):

    @property
    @abstractmethod
    def workers(self):
        raise NotImplementedError("Property should contain the number of workers to be called for data generation.")

    @property
    @abstractmethod
    def multiprocessing(self):
        raise NotImplementedError("Property should contain a boolean to tell the usage of multiprocessing for datageneration.")

    @property
    @abstractmethod
    def gpus(self):
        raise NotImplementedError("Property should contain the number of gpus to use.")

    @property
    @abstractmethod
    def project_name(self):
        raise NotImplementedError("Property should contain the project name.")

    @property
    @abstractmethod
    def workspace(self):
        raise NotImplementedError("Property should contain the workspace name.")

    @property
    @abstractmethod
    def weights(self):
        raise NotImplementedError("Property should contain the path to the weights to load inside the network. None if none to load.")

    @property
    @abstractmethod
    def network(self):
        raise NotImplementedError("Property should contain the network to be used.")

    @property
    @abstractmethod
    def epochs(self):
        raise NotImplementedError("Property should contain the number of epochs.")

    @property
    @abstractmethod
    def batch_size(self):
        raise NotImplementedError("Property should contain the batch size.")

    @property
    @abstractmethod
    def steps_per_epoch(self):
        raise NotImplementedError("Property should contain the number of steps per epochs.")

    @property
    @abstractmethod
    def optimizer(self):
        raise NotImplementedError("Property should contain the optimizer to use.")

    @property
    @abstractmethod
    def loss(self):
        raise NotImplementedError("Property should contain the loss function to use for training.")

    @property
    @abstractmethod
    def metrics(self):
        raise NotImplementedError("Property should contain the metrics.")

    @property
    @abstractmethod
    def callbacks(self):
        raise NotImplementedError("Property should contain all the callback functions.")

    @property
    @abstractmethod
    def train_generator(self):
        raise NotImplementedError("Property should contain the train generator. To be initialized with prepare_training_generators().")

    @property
    @abstractmethod
    def validation_generator(self):
        raise NotImplementedError("Property should contain the validation generator. To be initialized with prepare_training_generators().")

    @property
    @abstractmethod
    def test_generator(self):
        raise NotImplementedError("Property should contain the test generator. To be  initialized with the prepare_testing_generators().")

    @property
    @abstractmethod
    def evaluator(self):
        raise NotImplementedError("Property should contain the evaluator for the network.")

    @property
    @abstractmethod
    def displayer(self):
        raise NotImplementedError("Property should contain the displayer for the network.")
        
    @abstractmethod
    def prepare_for_inference(self):
        raise NotImplementedError("Should prepare the network for inference, to do nothing is the network is the same at training and inference time.")

    @abstractmethod
    def prepare_training_generators(self):
        raise NotImplementedError("Should prepare the generators for training, i.e the train and validation generators. This function is here to avoid loading unnecessary stuff.")

    @abstractmethod
    def prepare_testing_generator(self):
        raise NotImplementedError("Should prepare the generator for testing, i.e the test generator. This function is here to avoid loading unnecessary stuff.")

    @abstractmethod
    def prepare_evaluator(self):
        raise NotImplementedError("Should prepare the evaluator for testing. This function is here to avoid loading unnecessary stuff.")
