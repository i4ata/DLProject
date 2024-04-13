import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import EnWik8
from transformer import DecoderOnlyAFT

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyAFT(
        layers=12,
        e_dim=512,
        hid_dim=512,
        vocab_size=37,
        sequence_len=1024
    ).to(device)
    enwik = EnWik8(sequence_length=1024)
    enwik.split()
    enwik.get_dataloaders(batch_size=128)
    train_dataloader, val_dataloader = enwik.train_dataloader, enwik.val_dataloader

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=.5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(100)):

        train_loss, val_loss = 0, 0

        model.train()
        for X, y in train_dataloader:

            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        for X, y in val_dataloader:

            X, y = X.to(device), y.to(device)
            
            with torch.inference_mode():
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

            val_loss += loss.item()

        print(f'{epoch}: Train loss: {train_loss :.3}, Val loss: {val_loss :.3}')