from abc import abstractmethod

import numpy as np

from keras.utils import Sequence
from keras.utils import to_categorical

class TemplateGenerator(Sequence):
    """A generator for the pipeline given as template, the template is assured to work if the generator follows this example."""

    @property
    @abstractmethod
    def number_of_data_samples(self):
        raise NotImplementedError("A generator should be able to tell the number of data available.")

    @property
    @abstractmethod
    def shuffle(self):
        raise NotImplementedError("A generator should be able to shuffle its data if required.")

    @property
    @abstractmethod
    def batch_size(self):
        raise NotImplementedError("The batch size should be defined for a generator.")

    @abstractmethod
    def get_raw_input_label(self, index):
        """ Should return the raw input at a given batch index, i.e something displayable.
        
        # Argument:
            - index: The index of the batch
        """
        raise NotImplementedError("Should return the raw input at a given batch index, i.e something displayable.")

    @abstractmethod
    def __len__(self):
        """ Should return the number of batch per epoch."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """ Should return the batch target of the index specified.
        
        # Argument:
            - index: The index of the batch
        """
        raise NotImplementedError("Should return the batch target of the index specified.")

    @abstractmethod
    def get_batch_data(self, index):
        """ Should return the data associated for the batch specified if any. Should return None else.
        
        # Argument:
            - index: The index of the batch
        """
        raise NotImplementedError("Should return the data associated for the batch specified if any. Should return None else.")

    @abstractmethod
    def shuffle_batches(self):
        """ Should shuffle the batches of data."""
        raise NotImplementedError("Should shuffle the batches of data.")

    def on_epoch_end(self):
        """ To be called at the end of an epoch. The indexes of the data could be shuffle here."""
        pass
