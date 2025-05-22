# Server-initiated HE FL
This repo is used to provide advanced privacy protection in Federated Learning using Server-initiated Homomorphic Encryption. Individual training gradients are masked with a noise, Homomorphically Encrypted using CKKS, and are passed a long a chain of clients to create a secure aggregation system.

# Four main files
**main.py**: Contains code necessary to run simulation <br>
**network_node.py**: Conatins the NetworkSimulationClass which acts as the network in this simulation and transmits messages between the Clients and the Central Server <br>
**client_node.py**: Contains the ClientNodeClass which acts as a Client in the simulation <br>
**server_node.py**: Contains the ServerNodeClass which acts as the Central Server in the simulation <br>

# Python and Packages version
Python TenSEAL library required for CKKS file, current setup uses Python Version 3.9 to properly install all requirements. More up to date python versions may cause TenSEAL library to not install properly. Requirements.txt states to use numpy 2.0.1. If this doesn't work, use numpy 1.26.4

# Running Simulation
To run our program, run 
```
python main.py
```

Our program allow multiple configs
```
--dataset [dataset used]
--model [model type]
--alpha [data distribution]
--num_users [number of clients]
--local_ep [number of local epochs]
--partition_sizes [minimum size of the client route partitions]
--output_directory [output directory for results]
```

Example:
```python
python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=10 --output_directory=MNIST_baseline
```

# Standard FL implementation (FedAVG)
This is the implementation of the standard/basic FedAVG FL algorithm and is used to compare against proposed FL scheme. It takes the same arguements as above except for `--partition_size`
