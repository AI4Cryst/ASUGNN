### DataProcessing.py

`data_processing.py` is designed to extract structural data from a source database and write it into a target database. All screened structures are saved as a new database for use in neural network workflows. The specific operations performed are as follows:

1. **Extracting Data from the Database**: Connect to the source database and retrieve information for each entry, including atomic structures and formation energy.

2. **Processing Data**: Utilize the `Crylearn` module from the Crylearn library to convert the extracted atomic structure data into a graph representation. This includes generating node embeddings, adjacency matrices, and global information.

3. **Parallel Processing**: Employ `ProcessPoolExecutor` to process each database entry in parallel, significantly speeding up the data processing workflow.

- `--database_path`: the crysal saved in db.
- `--new_database_path`: the parsed graph data.


You can now run the script from the command line like this:

```bash
python data_processing.py --database_path /home/cb/cb_crystal/test_ASUnet/temp/structures.db --new_database_path ./temp/filter_self_struc_cif.db
```

### Model.py

`models.py` contains the implementation of the ASUGNN (Asymmetric Unit-Based Graph Neural Network) using PyTorch. The main components of this module are as follows:

1. **NetConfig**: A configuration class that sets up the base parameters for the ASUGNN model, allowing for easy adjustments and experimentation.

2. **CausalSelfAttention**: A module that implements a causal self-attention mechanism with learnable adjacency matrix projections, enabling the model to focus on relevant parts of the graph during learning.

3. **CausalCrossAttention**: A module designed to implement causal cross-attention, utilizing both graph and node features to enhance the model's understanding of relationships.

4. **ASU_codec_block**: A codec block that integrates self-attention, cross-attention, and multi-layer perceptron (MLP) modules, forming the building blocks of the ASUGNN architecture.

5. **ASU_Codec**: A stack of `ASU_codec_block` modules that forms the core of the attention-based encoder, facilitating complex feature extraction and representation learning.


### predict.py

`predict.py` is designed to load a pre-trained ASUGNN model, perform predictions on a dataset of graph-based data, and retrieve essential graph-related information from a database. The module allows users to quickly generate predictions without needing to retrain the model.

- `--db_path`: for the database path.
- `--model_path`: for the model path.


You can now run the script from the command line like this:

```bash
python predict.py --db_path /home/cb/cb_crystal/test_ASUnet/filter_self_struc_cif.db --model_path ASUGNN.pt
```



### Pre-trained Model

You can download the pre-trained ASUGNN model from the following link: [ASUGNN on Hugging Face](https://huggingface.co/caobin/ASUGNN)


