import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from typing import Tuple

class EnWik8:
    """Main class for the dataset"""

    def __init__(self, txt_dir: str = 'data/enwik8_clean.txt', sequence_length: int = 1024, dataset_fraction: float = .01) -> None:
        dataset = TorchDataset(txt_dir=txt_dir, sequence_length=sequence_length)
        indices = torch.randperm(len(dataset))[:int(dataset_fraction * len(dataset))]
        self.dataset = Subset(dataset=dataset, indices=indices)

    def split(self, lengths: Tuple[float, float, float] = (.8, .1, .1)) -> None:
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(dataset=self.dataset, lengths=lengths)

    def get_dataloaders(self, batch_size: int = 128, num_workers: int = 1) -> None:
        
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

class TorchDataset(Dataset):
    
    def __init__(self, txt_dir: str = 'data/enwik8_clean.txt', sequence_length: int = 1024) -> None:
        super().__init__()
        with open(txt_dir) as f:
            self.text = f.read()
        self.text = self.text.replace('\n', ' ')
        self.sequence_length = sequence_length
        self.unique_characters = sorted(set(self.text))
        self.len_unique_characters = len(self.unique_characters)
        
    def __len__(self) -> int:
        return len(self.text) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Adjust the index so that each element has full context
        index += self.sequence_length
        
        # Get the text that is the last 'sequence_length' characters before index
        context = self.text[:index][-self.sequence_length:]

        # The target is the single next character
        target = self.text[index]

        # Replace each character with its index in the vocabulary
        context = torch.tensor(list(map(self.unique_characters.index, context)))
        target = torch.tensor(self.unique_characters.index(target))

        return context, target
    