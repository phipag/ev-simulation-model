from abc import ABCMeta, abstractmethod


class Exporter(metaclass=ABCMeta):
    @abstractmethod
    def export(self, destination: str) -> None:
        ...
