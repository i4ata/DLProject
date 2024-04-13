import torch
import torch.nn as nn
from tqdm import tqdm
import pickle 

from dataset import EnWik8
from transformer import DecoderOnlyAFT
from early_stopper import EarlyStopper

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyAFT(
        layers=3,
        e_dim=64,
        hid_dim=64,
        vocab_size=37,
        sequence_len=256
    ).to(device)
    enwik = EnWik8(sequence_length=256)
    enwik.split()
    enwik.get_dataloaders(batch_size=128)
    train_dataloader, val_dataloader = enwik.train_dataloader, enwik.val_dataloader
    print(f'Batches in train_dataloder: {len(train_dataloader)}')
    print(f'Batches in val_dataloder: {len(val_dataloader)}')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=.5)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper()

    metrics = {
        'train': {
            'losses': [],
            'accuracies': []
        },
        'val': {
            'losses': [],
            'accuracies': []
        }
    }

    for epoch in tqdm(range(100)):
        train_loss, val_loss, train_acc, val_acc = 0, 0, 0, 0

        # Train loop
        model.train()
        for X, y in train_dataloader:

            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (y_pred.argmax(-1) == y).sum().item()

        # Validation loop
        model.eval()
        for X, y in val_dataloader:

            X, y = X.to(device), y.to(device)
            
            with torch.inference_mode():
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

            val_loss += loss.item()
            val_acc += (y_pred.argmax(-1) == y).sum().item()

        # Do early stopping
        if early_stopper.check(val_loss):
            print(f'Model stopped early due to risk of overfitting')
            break

        if early_stopper.save_model:
            print('Saving model')
            torch.save(model.state_dict(), 'models/model1.pt')

        # Aggregate metrics
        train_loss, val_loss = train_loss / len(train_dataloader), val_loss / len(val_dataloader)
        train_acc, val_acc = train_acc / len(train_dataloader), val_acc / len(val_dataloader)
        print(f'{epoch}: Train loss: {train_loss :.3}, Train acc: {train_acc :.3}, Val loss: {val_loss :.3}, Val acc: {val_acc :.3}')

        # Register metrics
        metrics['train']['losses'].append(train_loss)
        metrics['train']['accuracies'].append(train_acc)
        metrics['val']['losses'].append(val_loss)
        metrics['val']['accuracies'].append(val_acc)

    # Save the metrics
    with open('metrics1.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # Test best model
    model.load_state_dict(torch.load('models/model1.pt'))
    test_loss, test_acc = 0, 0
    for X, y in enwik.test_dataloader:

        X, y = X.to(device), y.to(device)
        
        with torch.inference_mode():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        test_loss += loss.item()
        test_acc += (y_pred.argmax(-1) == y).sum().item()
    print(test_loss / len(enwik.test_dataloader), test_acc / len(enwik.test_dataloader))