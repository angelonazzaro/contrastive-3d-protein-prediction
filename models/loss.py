import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, use_sigmoid: bool = False):
        """
        Contrastive Loss initialization.

        Args:
            use_sigmoid (bool): If True, use the Sigmoid loss instead of CLIP loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = torch.nn.Parameter(0.07 if not use_sigmoid else torch.log(10))
        self.use_sigmoid = use_sigmoid

        if use_sigmoid: 
            self.bias = torch.nn.Parameter(-10)


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

        if self.use_sigmoid:
            # following https://arxiv.org/pdf/2303.15343.pdf pseudo-code implementation
            logits = torch.matmul(graph_embeddings, dna_embeddings.t()) * self.temperature + self.bias
            labels = 2 * torch.eye(len(logits)) - torch.ones(batch_size)
            loss = -F.log_sigmoid(labels * logits).sum() / len(logits)
        else: 
            # Compute the similarity matrix (dot product)
            logits_per_graph = torch.matmul(graph_embeddings, dna_embeddings.t()) * torch.exp(self.temperature)
            labels = torch.arange(len(logits_per_graph), device=logits_per_graph.device)
            
            graph_loss = F.cross_entropy(logits_per_graph, labels)
            dna_loss = F.cross_entropy(logits_per_graph.t(), labels)
            
            loss = (graph_loss + dna_loss) / 2.0

        return {"loss": loss, "labels": labels, "logits": logits_per_graph}