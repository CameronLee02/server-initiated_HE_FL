import torch.nn.functional
from torch.utils.data import DataLoader
import torch

def test_fun(net, dataset, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.local_bs, shuffle=False)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = net(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy, test_loss