import torch
from abc import ABC, abstractmethod


class Scheduler(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update_parameter(
            self
    ):
        pass
