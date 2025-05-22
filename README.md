# Server-initiated HE FL
This repo is used to provide advanced privacy protection in Federated Learning using Server-initiated Homomorphic Encryption. Individual training gradients are masked with a noise, Homomorphically Encrypted using CKKS, and passed a long a chain of clients to create a secure aggregation system.

# Four main files
**main.py**: Contains code necessary to run simulation <br>
**network_node.py**: Contains the NetworkSimulationClass, which acts as the network in this simulation and transmits messages between the Clients and the Central Server <br>
**client_node.py**: Contains the ClientNodeClass, which acts as a Client in the simulation <br>
**server_node.py**: Contains the ServerNodeClass, which acts as the Central Server in the simulation <br>

# Python and Packages version
Python TenSEAL library is required for CKKS and the current setup uses Python Version 3.9 to properly install all requirements. More up to date python versions will cause issues with the TenSEAL library. Requirements.txt states to use numpy 2.0.1. If this doesn't work, use numpy 1.26.4

# Running Simulation
To start the simulation, run 
```
python main.py
```

Our program allows multiple configs
```
--dataset [dataset used]
--model [model type]
--alpha [data distribution]
--num_users [number of clients]
--local_ep [number of local epochs]
--partition_sizes [minimum number of the client in each route partition]
--output_directory [output directory for results]
```

Example:
```
python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=10 --output_directory=MNIST_baseline
```

# Standard FL implementation (FedAVG)
This is the implementation of the standard/basic FedAVG FL algorithm and is used to compare against our proposed FL scheme. It takes the same arguments as above, except for `--partition_size`

Example:
```
python standard_fl_implementation.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_standard_fl
```

# Results
The simulation captures information about each training epoch and records them in the specified output directory. 

Four CSV files are created: <br>
**scores**: records the model accuracy and loss after each training epoch
**sizes**: records the size of the gradients being encrypted and decrypted
**times**: records the individual runtime of each procedure in the training epoch
**transmissions**: records the number of transmissions done during each training epoch
