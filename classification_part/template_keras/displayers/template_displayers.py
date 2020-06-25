from abc import ABC, abstractmethod

class TemplateDisplayer(ABC):

    @abstractmethod
    def display(self, predictions, inputs):
        """ Displays the predictions on the given inputs.
        
        # Arguments:
            - predictions: The predictions output by the predict function of keras.
            - inputs: The inputs corresponding to the predictions.
        """

        raise NotImplementedError("Should displays the predictions on the given inputs.")

    @abstractmethod
    def display_with_gt(self, predictions, inputs, groundtruth):
        """ Displays the predictions and the groundtruths on the given inputs.
        
        # Arguments:
            - predictions: The predictions output by the predict function of keras.
            - inputs: The inputs corresponding to the predictions.
            - groundtruth: The groundtruths corresponding to the predictions.
        """
        raise NotImplementedError("Should displays the predictions and the groundtruths on the given inputs.")
