import torch
from torch import nn, device as torch_device
from torch.utils.tensorboard import SummaryWriter
from models.loss import ContrastiveLoss
from transformers import AutoTokenizer, AutoModel, BertConfig
from models.constants import GRAPH_MODELS
from torch_geometric.graphgym import load_ckpt, save_ckpt, remove_ckpt, clean_ckpt


class C3DPNet(nn.Module):
    def __init__(self,
                 bio_model: str = "zhihan1996/DNABERT-2-117M",
                 bio_tokenizer: str = "zhihan1996/DNABERT-2-117M",
                 graph_model: str = "GraphSAGE",
                 **kwargs):
        super().__init__()

        config = BertConfig.from_pretrained(bio_model)
        self.bio_tokenizer = AutoTokenizer.from_pretrained(bio_tokenizer)
        self.bio_model = AutoModel.from_config(config)
        self.graph_model = GRAPH_MODELS[graph_model](**kwargs)
        self.loss = ContrastiveLoss()
        self.device = torch_device('cpu', 0)

    def forward(self):
        pass

    def to(self, device):
        m = super().to(device)
        self.device = next(m.parameters()).device
        return m


    def log_metrics(self, loss: float, accuracy: float, global_step: int) -> None:
        # Log metrics in TensorBoardX
        self.writer.add_scalar('Loss', loss, global_step=global_step)
        self.writer.add_scalar('Accuracy', accuracy, global_step=global_step)

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            # Other parameters
        }
        save_ckpt(checkpoint, epoch)

    def load_checkpoint(self, epoch: int) -> None:
        checkpoint = load_ckpt(epoch)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load other parameters if needed

    def remove_checkpoint(self, epoch: int) -> None:
        # Remove the model checkpoint
        remove_ckpt(epoch)

    def clean_checkpoints(self) -> None:
        # Remove all checkpoints except the last one
        clean_ckpt()

    def train_step(self, data: torch.Tensor, epoch: int, batch_idx: int,
                   train_loader: torch.utils.data.DataLoader) -> None:
        # ...

        # Calculate and log metrics
        # loss_value = calculate_loss(predictions, labels)
        # accuracy_value = calculate_accuracy(predictions, labels)

        # Call the method to log metrics
        global_step = epoch * len(train_loader) + batch_idx
        self.log_metrics(loss_value, accuracy_value, global_step)

        # Save the model checkpoint at each epoch
        self.save_checkpoint(epoch)
        # You can also remove old checkpoints if desired

        # Call the method to remove old checkpoints
        self.clean_checkpoints()