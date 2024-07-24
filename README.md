# ASUGNN
Asymmetric Unit-Based Graph Neural Network for Crystal Property Prediction

![Screenshot 2024-07-24 at 19 24 25](https://github.com/user-attachments/assets/ecd5c325-a1a6-49f1-9f41-c4fc4aa48c1f)

## Dataprocessing.py

`Dataprocessing.py` is used to extract structural data from one database and write it into another database. This process is accelerated through parallel processing. The specific operations are as follows:

1. **Extracting Data from the Database**: Connect to the source database and retrieve information for each entry, including atomic structures and formation energy.
2. **Processing Data**: Use the `cry2graph` module from the Crylearn library to convert the extracted atomic structure data into a graph representation, including node embeddings, adjacency matrices, and global information.
3. **Parallel Processing**: Use `ProcessPoolExecutor` to process each database entry in parallel to speed up the processing.

## Model.py

`Model.py` contains the implementation of ASUGNN using PyTorch. The main components are:


1. **NetConfig**: A configuration class for setting up the base parameters for the ASUGNN model.
2. **CausalSelfAttention**: A module that implements a causal self-attention mechanism with learnable adjacency matrix projections.
3. **CausalCrossAttention**: A module for implementing causal cross-attention, utilizing graph and node features.
4. **ASU_codec_block**: A codec block that combines self-attention, cross-attention, and multi-layer perceptron (MLP) modules.
5. **ASU_Codec**: A stack of `ASU_codec_block` modules forming the core of the attention-based encoder.


## predict.py

Download the pretrained ASUGNN at : [ASUGNN](https://huggingface.co/caobin/ASUGNN)

`predict.py` is designed to load a pre-trained model, perform predictions on a dataset of graph-based data, and retrieve essential graph-related information from a database.

Example:
```python
model_path = 'model.pt'
dataset = ...  # Replace this with your dataset instance
pred, tru = load_model_and_predict(model_path, dataset)
```
