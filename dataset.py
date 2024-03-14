import torch
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Tuple

class EnWik8:
    """Main class for the dataset"""

    def __init__(self, txt_dir: str = 'data/enwik8_clean.txt', sequence_length: int = 1024) -> None:
        self.dataset = TorchDataset(txt_dir=txt_dir, sequence_length=sequence_length)

    def split(self, lengths: Tuple[float, float, float] = (.8, .1, .1)) -> None:
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(dataset=self.dataset, lengths=lengths)

    def get_dataloaders(self, batch_size: int = 128, num_workers: int = 1, pin_memory: bool = False):
        
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

class TorchDataset(Dataset):
    
    def __init__(self, txt_dir: str = 'data/enwik8_clean.txt', sequence_length: int = 1024) -> None:
        super().__init__()
        with open(txt_dir) as f:
            self.text = f.read()
        self.sequence_length = sequence_length
        self.unique_characters = sorted(set(self.text))
        self.character_to_index = {character : i for i, character in enumerate(self.unique_characters)}

    def __len__(self) -> int:
        return len(self.text) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Adjust the index so that each element has full context
        index += self.sequence_length
        
        # Get the text that is the last 'sequence_length' characters behind index
        context = self.text[:index][-self.sequence_length]

        # The target is the single next character
        target = self.text[index]

        # One hot encode them
        one_hot_encoded_context = torch.zeros(self.sequence_length, len(self.unique_characters))
        for i, character in enumerate(context):
            one_hot_encoded_context[i, self.character_to_index[character]] = 1
        one_hot_encoded_target = torch.zeros(len(self.unique_characters))
        one_hot_encoded_target[self.character_to_index[target]] = 1
    
        return one_hot_encoded_context, one_hot_encoded_target
    
if __name__ == '__main__':
    d = EnWik8()
    print('dataset initialized')
    d.split()
    print('dataset split into train-val-test')
    d.get_dataloaders()
    print('dataloaders initialized')
    
    train_sample_X, train_sample_Y = next(iter(d.train_dataloader))
    print(train_sample_X.shape, train_sample_Y.shape)
    
    val_sample_X, val_sample_Y = next(iter(d.val_dataloader))
    print(val_sample_X.shape, val_sample_Y.shape)
    
    test_sample_X, test_sample_Y = next(iter(d.test_dataloader))
    print(test_sample_X.shape, test_sample_Y.shape)
    