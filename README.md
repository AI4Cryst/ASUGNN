
# [ASUGNN](https://doi.org/10.1107/S1600576724011336)

**Asymmetric Unit-Based Graph Neural Network for Crystal Property Prediction**

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
- `Crylearn`
- `ase`

### Example Usage

Here is an example of how to use the `Crylearn` package to extract graph embedding data from a crystal database:

```python
from Crylearn import cry2graph
from ase.db import connect

# Connect to the demo database
database = connect('demo.db')
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


## Toolkit sharing 🚀 🚀 🚀
### **ASUkit: Symmetric Graph Representation of a Crystal**

ASUkit provides a Symmetric graph-based representation of a crystal structure. see tutorial [here](https://github.com/AI4Cr/ASUGNN/blob/main/tutorial/tutorial.ipynb) 

### **Concept**
First, the package decomposes the crystal structure into its fundamental unit: the **Asymmetric Unit (ASU)**, ref definition at [RCSB PDB guide](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies).

### **Outputs**
ASUkit generates the following key outputs:

- **`node`**: An \(N \times 106\) matrix, where \(N\) is the number of atoms in the conventional unit cell, and 106 represents the atomic and structural attributes (refer to our paper for details).
- **`asu_adj_matrix`**: An \(N \times N\) adjacency matrix, where atoms within the same symmetry group are connected by 1, and all other connections are 0.
- **`distance_matrix`**: An \(N \times N\) matrix storing pairwise atomic distances in Cartesian coordinates.
- **`ideal_pxrd`**: A simulated ideal powder X-ray diffraction (PXRD) pattern, reflecting the atomic arrangement in reciprocal space.

### **References**
- *ASUGNN: An asymmetric-unit-based graph neural network for crystal property prediction.*  
  *Applied Crystallography, 58(1).*  
  [Link to paper](https://journals.iucr.org/paper?ei5123)

- **GitHub Repository**:  
  [ASUGNN GitHub](https://github.com/AI4Cr/ASUGNN/tree/main/paper)


