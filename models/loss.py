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

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, return_details: bool = False):
        """
        Forward pass of the Contrastive Loss.

        Args:
            z1 (torch.Tensor): Representations from the first set of samples.
            z2 (torch.Tensor): Representations from the second set of samples.
            return_details (bool): If True, return a dictionary with loss, labels, and logits.

        Returns:
            torch.Tensor or dict: Contrastive loss value or a dictionary containing loss, labels, and logits.
        """
        # Normalize representations
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        # Concatenate representations to form positive and negative batches
        representations = torch.cat([z1, z2], dim=1)

        # Compute the similarity matrix (dot product)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Build target labels
        labels = torch.arange(0, representations.size(0)).to(z1.device)
        labels = labels.unsqueeze(0)
        labels = labels.repeat(labels.size(1), 1)

        # Mask diagonals in the label matrix
        mask = torch.eq(labels, labels.T).float()

        # Compute contrastive loss
        positive_samples = similarity_matrix[~mask.bool()].view(representations.size(0), -1)
        negative_samples = similarity_matrix[mask.bool()].view(representations.size(0), -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(logits.size(0)).to(z1.device).long()
        
        if self.use_sigmoid:
            # Use sigmoid function
            sigmoid_output = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(sigmoid_output, labels.float())
        else:
            # Use cross entropy
            loss = F.cross_entropy(logits, labels)

        if return_details:
            return {"loss": loss, "labels": labels, "logits": logits}
        else:
            return loss
