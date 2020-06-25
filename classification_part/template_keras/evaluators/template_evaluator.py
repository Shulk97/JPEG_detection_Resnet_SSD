from abc import ABC, abstractmethod

class TemplateEvaluator(ABC):

    @abstractmethod
    def __str__(self):
        """ Should print the results after the call to this class. """
        raise NotImplementedError("Should print the results after the call to this class.")

    @abstractmethod
    def __call__(self, model, test_generator=None):
        """ Should calculate the evaluation for a given generator.
        
        # Arguments:
            - model: The model to use for predictions.
            - test_generator: The generator to use, can be None if the generator was initialized with the init function.
        """
        raise NotImplementedError("Should calculate the evaluation for a given generator.")

    @abstractmethod
    def make_runs(self, model, test_generator=None, number_of_runs=10):
        """ Should evaluate on multiple runs. Should also display the time for the run as this is used in the time inference script. 
        # Arguments:
            - model: The model to use for predictions.
            - test_generator: The generator to use, can be None if the generator was initialized with the init function.
            - number_of_run: The number of run that should be carried.
        
        """
        raise NotImplementedError("Should evaluate on multiple runs. Should also display the time for the run as this is used in the time inference script. ")
    
    @abstractmethod
    def display_results(self):
        """ Should display the results of an evaluation, what ever they are. To be run after a call to the Evaluator."""
        raise NotImplementedError("Should display the results of an evaluation, what ever they are. To be run after a call to the Evaluator.")

    @property
    @abstractmethod
    def test_generator(self):
        raise NotImplementedError("Should contain the generator to be used.")
