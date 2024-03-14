import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import EnWik8

class DummyModel(nn.Module):
    def __init__(self, sequence_length: int = 1024) -> None:
        super(DummyModel, self).__init__()
        self.unique_characters = 38
        self.fc = nn.Linear(sequence_length * self.unique_characters, self.unique_characters)
    
    def forward(self, x: torch.Tensor):
        return self.fc(torch.flatten(x, start_dim=1))
    
if __name__ == '__main__':
    dataset = EnWik8()
    dataset.split()
    dataset.get_dataloaders()
    X, y = next(iter(dataset.train_dataloader))

    model = DummyModel()
    y_pred = model(X)
    print(y_pred.shape)
    print(F.cross_entropy(y_pred, y))