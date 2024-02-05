import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5, use_sigmoid: bool = False):
        """
        Contrastive Loss initialization.

        Args:
            temperature (float): A hyperparameter controlling the scale of the similarity.
            use_sigmoid (bool): If True, use the sigmoid function instead of cross entropy.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_sigmoid = use_sigmoid

    def forward(self, graph_embeddings: torch.Tensor, dna_embeddings: torch.Tensor):
        """
        Forward pass of the Contrastive Loss.

        Args:
            z1 (torch.Tensor): Representations from the first set of samples.
            z2 (torch.Tensor): Representations from the second set of samples.

        Returns:
            torch.Tensor or dict: Contrastive loss value or a dictionary containing loss, labels, and logits.
        """
        # Normalize representations
        graph_embeddings = F.normalize(graph_embeddings, dim=1, p=2)
        dna_embeddings = F.normalize(dna_embeddings, dim=1, p=2)

        # Compute the similarity matrix (dot product)
        logits_per_graph = torch.matmul(graph_embeddings, dna_embeddings.t()) / self.temperature
        labels = torch.arange(len(logits_per_graph), device=logits_per_graph.device)

        graph_loss = self.__loss(logits_per_graph, labels)
        dna_loss = self.__loss(logits_per_graph.t(), labels)

        return {"loss": (graph_loss + dna_loss) / 2.0, "labels": labels, "logits": logits_per_graph}

    def __loss(self, logits, labels): 
        if self.use_sigmoid:
            # Use sigmoid function
            sigmoid_output = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(sigmoid_output, labels.float())
        else:
            # Use cross entropy
            loss = F.cross_entropy(logits, labels)
        return loss