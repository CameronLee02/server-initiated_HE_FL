import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch, args):
    token_tensors, labels = zip(*batch)
    lengths = torch.tensor([t.size(0) for t in token_tensors], dtype=torch.long)
    token_tensors = [t[:args.max_seq_len] for t in token_tensors]
    padded = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    lengths = torch.clamp(lengths, max=args.max_seq_len)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels
    

def test_fun(net, dataset, args):    
    net.to(args.device)
    net.eval()
    test_loss = 0
    correct = 0
    if args.model == 'lstm':
        data_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=False, collate_fn=lambda b: collate_batch(b, args))
    else:
        data_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=False)

    with torch.no_grad():
        for batch in data_loader:
            if args.model == 'lstm':
                inputs, lengths, labels = batch
                inputs, lengths, labels = inputs.to(args.device), lengths.to(args.device), labels.to(args.device)
                outputs = net(inputs, lengths)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = net(inputs)
                

            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="sum")
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    return accuracy, test_loss

