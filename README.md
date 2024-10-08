
# ASUGNN

**Asymmetric Unit-Based Graph Neural Network for Crystal Property Prediction**

![ASUGNN](https://github.com/user-attachments/assets/ecd5c325-a1a6-49f1-9f41-c4fc4aa48c1f)

---

## Overview

ASUGNN is a Graph Neural Network (GNN) designed for crystal property prediction, leveraging asymmetric unit-based graph representations. This approach efficiently captures the structural and chemical properties of crystals, making it highly suitable for materials informatics tasks.


+ HMdataset.db: A zero-shot test dataset
You can test your model's zero-shot capability by predicting the formation energy per atom using the HMdataset (1,100 data) and upload your results on Kaggle : https://www.kaggle.com/competitions/asugnn.


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


