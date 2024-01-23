import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        """
        Contrastive Loss initialization.

        Args:
            temperature (float): A hyperparameter controlling the scale of the similarity.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Forward pass of the Contrastive Loss.

        Args:
            z1 (torch.Tensor): Representations from the first set of samples.
            z2 (torch.Tensor): Representations from the second set of samples.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Normalize representations
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        # Concatenate representations to form positive and negative batches
        representations = torch.cat([z1, z2], dim=0)

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

        loss = F.cross_entropy(logits, labels)

        return loss


if __name__ == "__main__":
    # Create sample data
    batch_size = 8
    embedding_size = 10

    z1 = torch.randn(batch_size, embedding_size)
    z2 = torch.randn(batch_size, embedding_size)

    # Initialize the ContrastiveLoss
    contrastive_loss = ContrastiveLoss(temperature=0.5)

    # Forward pass
    loss = contrastive_loss(z1, z2)

    # Print the loss
    print("Contrastive Loss:", loss.item())
