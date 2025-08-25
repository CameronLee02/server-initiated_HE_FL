import torch
from torch.utils.data import DataLoader, Dataset
from models.test import collate_batch

class LocalUpdateCNN:
    def __init__(self, args, dataset, idxs):
        self.args = args
        split_dataset = DatasetSplitCNN(dataset, idxs)
        self.ldr_train = DataLoader(split_dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                outputs = net(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class DatasetSplitCNN(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

    
class LocalUpdateIMDB:
    def __init__(self, args, dataset, idxs, vocab, tokenizer):
        self.args = args
        split_dataset = DatasetSplitIMDB(dataset, idxs, vocab, tokenizer, self.args.max_seq_len)
        self.ldr_train = DataLoader(split_dataset, batch_size=self.args.local_bs, shuffle=True, collate_fn=lambda b: collate_batch(b, args))

    def train(self, net):
        net = net.to(self.args.device)
        net.train()
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for inputs, lengths, labels in self.ldr_train:
                inputs, lengths, labels = inputs.to(self.args.device), lengths.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad(set_to_none=True)
                net.zero_grad()
                outputs = net(inputs, lengths)   # assume LSTM takes (inputs, lengths), like IMDB Dataset
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class DatasetSplitIMDB(Dataset):
    def __init__(self, dataset, idxs, vocab=None, tokenizer=None, max_seq_len=256):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        ex = self.dataset[self.idxs[item]]
        text = ex["text"]
        label = ex["label"]

        if self.tokenizer is not None and self.vocab is not None:
            tokens = self.tokenizer(text)
            token_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
            token_ids = torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long)
        else:
            token_ids = torch.tensor([], dtype=torch.long)

        return token_ids, int(label)