import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="Global rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users in FL: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of training samples selected from each local training set")
    parser.add_argument('--alpha', type=float, default=10, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=12, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--partition_size', type=int, default=10, help='minimum size of the partitions')
    parser.add_argument('--output_directory', type=str, default='results', help="the name of the directory to write the results to")
    parser.add_argument('--max_seq_len', type=int, default=256, help="the max length a sequence can be")
    parser.add_argument('--embed_dim', type=int, default=100, help="")
    parser.add_argument('--hidden_dim', type=int, default=128, help="")
    parser.add_argument('--max_vocab_size', type=int, default=20000, help="")

    # other arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help="name of dataset")
    parser.add_argument('--iid', default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args

def args_parser2():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="Global rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users in FL: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of training samples selected from each local training set")
    parser.add_argument('--alpha', type=float, default=10, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=12, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--output_directory', type=str, default='results', help="the name of the directory to write the results to")
    parser.add_argument('--max_seq_len', type=int, default=256, help="the max length a sequence can be")
    parser.add_argument('--embed_dim', type=int, default=100, help="")
    parser.add_argument('--hidden_dim', type=int, default=128, help="")
    parser.add_argument('--max_vocab_size', type=int, default=20000, help="")

    # other arguments
    parser.add_argument('--dataset', type=str, default='Synthetic', help="name of dataset")
    parser.add_argument('--iid', default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args

