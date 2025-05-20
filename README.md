

**ASUGNN ： Asymmetric Unit-Based Graph Neural Network for Crystal Property Prediction** | [Paper](https://doi.org/10.1107/S1600576724011336)

![image](https://github.com/user-attachments/assets/d2af74d4-4519-4386-b0cc-ebfb43f3b8f9)

---

## Overview

ASUGNN is a Graph Neural Network (GNN) designed for crystal property prediction, leveraging asymmetric unit-based graph representations. This approach efficiently captures the structural and chemical properties of crystals, making it highly suitable for materials informatics tasks.


+ HMdataset.db: A zero-shot test dataset
You can test your model's zero-shot capability by predicting the formation energy per atom using the [HMdataset](https://huggingface.co/caobin/ASUGNN/blob/main/README.md) (1,100 data) and upload your results on Kaggle : https://www.kaggle.com/competitions/asugnn.
+ Source Code : [src](https://github.com/AI4Cr/ASUGNN/tree/main/src)
+ Tutorial : [notebook](https://github.com/AI4Cr/ASUGNN/blob/main/Tutorial.ipynb)

## Crystal Data
If you need the organized crystal database, visit here: https://huggingface.co/datasets/caobin/CrystDB

## Graph Embedding of ASUGNN

The graph embedding is performed on each entry in the database using the `Crylearn` package. It extracts structural information about the crystal, including node embeddings, the ASU matrix, a distance matrix, and the simulated PXRD pattern.


### Requirements

- `Python 3.9.19`
- `ASUkit` [ref](https://github.com/AI4Cr/ASUGNN/blob/main/tutorial/tutorial.ipynb) 
- `ase`

### Example Usage  🚀 🚀 🚀

Here is an example of how to use the `ASUkit` package to extract graph embedding data from a crystal database:

```python
from ASUkit import cry2graph
from ase.db import connect

# Connect to the demo database
database ='demo.db'
entry_id = 1

# Parse the entry and extract the graph embedding
N, ASUAM, DAM, PXRD = cry2graph.parser(database, entry_id).get()
```

### Output

- **N** (`np.ndarray`): The node embeddings, where each node (atom) is represented by a 106-dimensional feature vector (shape: N x 106).
- **ASUAM** (`np.ndarray`): The Asymmetric Unit Adjacency Matrix (shape: N x N).
- **DAM** (`np.ndarray`): The distance matrix between nodes in Cartesian coordinates (shape: N x N).
- **PXRD** (`np.ndarray`): The simulated diffraction pattern of the crystal, capturing global structural information (shape: 140-dimensional).

### Explanation

- The **node embeddings** represent atomic features within the lattice.
- The **distance matrix** provides the distances between pairs of atoms.
- The **PXRD** vector represents the simulated diffraction pattern, offering a condensed representation of the crystal structure.

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.




