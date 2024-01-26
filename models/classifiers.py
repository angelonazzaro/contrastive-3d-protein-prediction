import abc
from abc import abstractmethod
from typing import Callable, Optional, Union, Iterable, Any

import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score


# original source: https://github.com/Attornado/protein-representation-learning/
class ClassificationLoss(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self._loss: Optional[Callable] = None

    def forward(self, targets: torch.Tensor, *outputs: torch.Tensor) -> torch.Tensor:
        """
        :param targets: labels
        :param outputs: predictions
        :return: loss value
        """
        outputs = outputs[0]
        loss = self._loss(outputs, targets)
        return loss

    def get_accuracy(self, targets: torch.Tensor, *outputs: torch.Tensor) -> float:
        outputs: torch.Tensor = outputs[0]
        acc = self._calculate_accuracy(outputs, targets)
        return acc

    @abstractmethod
    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        correct = self._get_correct(outputs)
        return float(100. * (correct == targets).sum().float() / targets.size(0))


class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, weights: Optional[Union[torch.Tensor, Iterable]] = None, reduction: Optional[str] = None,
                 label_smoothing: float = 0.0):
        super().__init__()

        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

        if reduction is not None:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weights,
                                                                              label_smoothing=label_smoothing)
        else:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=weights,
                                                                              label_smoothing=label_smoothing)

    @property
    def weights(self) -> Optional[torch.Tensor]:
        return self.__weights

    @weights.setter
    def weights(self, weights: Optional[Union[torch.Tensor, Iterable]]):
        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)


class GraphClassifier:
    def __init__(self, dim_features: int, dim_target: int, config: dict):
        """
        Generic graph classifier abstract class.

        :param dim_features: An integer representing the dimensionality of the input features for the graph classifier.
        :type dim_features: int
        :param dim_target: The dimensionality of the target variable, which is the variable we want to predict in the
            classification task.
        :type dim_target: int
        :param config: The `config` parameter is a dictionary that contains configuration settings for the
            `GraphClassifier` model.
        :type config: dict
        """
        super(GraphClassifier, self).__init__()
        self.__in_channels: int = dim_features
        self.__dim_target: int = dim_target
        self.__config_dict: dict = config

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @in_channels.setter
    def in_channels(self, in_channels: int):
        self.__in_channels = in_channels

    @property
    def dim_target(self) -> int:
        return self.__dim_target

    @dim_target.setter
    def dim_target(self, dim_target: int):
        self.__dim_target = dim_target

    @property
    def config_dict(self) -> dict:
        return self.__config_dict

    @config_dict.setter
    def config_dict(self, config_dict: dict):
        self.__config_dict = config_dict

    def test(self,
             y,
             y_hat: Optional[Any] = None,
             x: Optional[torch.Tensor] = None,
             edge_index: Optional[torch.Tensor] = None,
             batch_index: torch.Tensor = None,
             criterion: ClassificationLoss = MulticlassClassificationLoss(),
             top_k: Optional[int] = None,
             *args, **kwargs) -> (float, Optional[float], float, float, float, float):
        """
        Takes in a batch graph (or the predictions) and the corresponding labels, and returns the loss, accuracy, top-k
        accuracy, precision, recall, and F1-score.

        :param x: torch.Tensor = The node features
        :type x: torch.Tensor
        :param edge_index: The edge indices of the graph
        :type edge_index: torch.Tensor
        :param y: The target labels
        :param y_hat: The optional predicted labels
        :param batch_index: The batch index of the nodes
        :type batch_index: torch.Tensor
        :param criterion: The loss function to use
        :type criterion: Callable
        :param top_k: k for computing top_k accuracy, *args, **kwargs
        :type top_k: Optional[int]
        :return: The loss, accuracy, top-k accuracy, precision, recall, and F1-score.
        """

        # Get the number of classes
        n_classes = self.dim_target

        # Get predictions
        if y_hat is None:
            y_hat = self.forward(x.float(), edge_index, batch_index, *args, **kwargs)

        # Compute loss
        loss = self.loss(y_hat=y_hat, y=y, criterion=criterion)

        # Remove additional loss terms
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]

        # Compute the metrics
        acc = accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        if top_k is not None:
            top_k_acc = float(accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, top_k=top_k,
                                       average="macro"))
        else:
            top_k_acc = None
        prec = precision(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        rec = recall(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        f1 = f1_score(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")

        return float(loss), float(acc), top_k_acc, prec, rec, f1

    def loss(self,
             y,
             x: Optional[torch.Tensor] = None,
             edge_index: Optional[torch.Tensor] = None,
             batch_index: Optional[torch.Tensor] = None,
             y_hat: Optional[torch.Tensor] = None,
             criterion: ClassificationLoss = MulticlassClassificationLoss(),
             additional_terms: list[torch.Tensor] = None,
             *args, **kwargs) -> torch.Tensor:

        # If predictions are not given, compute them using the model
        if y_hat is None:
            y_hat = self.forward(x.float(), edge_index, batch_index, *args, **kwargs)

        # Compute loss with given criterion
        loss = criterion(y, y_hat)

        # Add pre-computed additional loss terms to the loss
        if additional_terms is not None:
            for additional_term in additional_terms:
                loss = loss + additional_term

        return loss

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        raise NotImplementedError(f"Each {self.__class__} class has to implement the forward() method")

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "dim_features": self.in_channels,
            "dim_target": self.dim_target,
            "config": self.config_dict
        }
