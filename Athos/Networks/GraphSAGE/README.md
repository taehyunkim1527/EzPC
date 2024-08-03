This folder contains code for the GraphSAGE. 

This folder contains GraphSAGE, which can be set to different sizes and aggregation methods.

## Setup
- To run training for example for the Lenet-Small network, execute the following: `python3 lenetSmall_mnist_train.py`.
- Subsquently to run inference, use this: `python3 lenetSmall_mnist_inference.py 1`, where `1` can be replaced by apt image number of MNIST. This command also dumps the TensorFlow metadata required for further compilation.
