import json
import torch
import glob
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import random
from ase.db import connect
from ase.db.row import AtomsRow
from dscribe.descriptors import SOAP
from scipy.stats import rankdata, norm
from joblib import Parallel, delayed




class MyCollator(object):
    def __init__(self, mysql_url, species):
        self.mysql_url = mysql_url
        self.species = species

    def __call__(self, examples):
        
        rows = query_database(self.mysql_url, examples)        
        
        # pos = [row.get('positions') for row in rows]
        ase_crystal_list = [row.toatoms() for row in rows]
        nodes = [item.get_atomic_numbers() for item in ase_crystal_list]
        energy = [item.data['energy'] for item in rows]


        distance_matrix = [item.get_all_distances(mic=True) for item in ase_crystal_list]
        # distance_matrix_trimmed = [threshold_sort(
        #     item,
        #     8.0,
        #     12,
        #     adj=False,
        # ) for item in distance_matrix]

        node_extra = [setGraphFea(rows[i], distance_matrix[i]) for i in range(0, len(rows))]

        distance_matrix_trimmed = [mask_elements_over_threshold(item, 8.0) for item in distance_matrix]

        distance_embd = [GBF_distance_encode(item, 0.0, 8.0, 12) for item in distance_matrix_trimmed]


        # make_feature_SOAP = SOAP(
        #     species=self.species,
        #     r_cut=8.0,
        #     n_max=6,
        #     l_max=4,
        #     sigma=0.3,
        #     periodic=True,
        #     sparse=False,
        #     average="inner",
        #     rbf="gto",
        # )

        # soap_fea = [make_feature_SOAP.create(item) for item in ase_crystal_list]
        # soap_fea = np.vstack(soap_fea)


        max_len = np.max([len(item) for item in nodes])


        max_seq_length = 444
        # Batch size limit for the maximum sequence length
        max_batch_size_for_max_seq_length = 50
        
        
        if max_len >= max_seq_length:
            batch_size_mini = max_batch_size_for_max_seq_length
        else:
            # Adjust the formula as needed to scale the batch size based on sequence length
            # This example linearly decreases the batch size from max_batch_size_for_max_seq_length
            # to 1 as the sequence length increases from 1 to max_seq_length.
            # Adjust this formula based on your specific memory constraints and experiment results.
            batch_size_mini = max(50, int(max_batch_size_for_max_seq_length * (max_seq_length / max_len)))


        # batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
        # processed_batches = []
        # for minibatch in batches:
        #     data = [item[0] for item in minibatch]
        #     # 填充序列以匹配批次中最长的序列
        #     data_padded = pad_sequence(data, batch_first=True)
        #     processed_batches.append(data_padded)


        batch_size = len(nodes)

        nodes_padded = np.zeros([batch_size, max_len])
        distance_padded = np.zeros([batch_size, max_len, max_len, 12])
        nodes_extra_padded = np.zeros([batch_size, max_len, 3 * 11 * 11])


        for i, item in enumerate(nodes):
            len_temp = len(item)
            nodes_padded[i, : len_temp] = item
            energy[i] /= len(item)
            distance_padded[i, :len_temp, :len_temp, :] = distance_embd[i]
            nodes_extra_padded[i, :len_temp, :] = node_extra[i]


        energy = [norm.cdf((np.arcsinh(item / 10.0) + 0.5) * 5.0) * 3.0 for item in energy]
        
        batches_idx = [[i,min(i + batch_size_mini, len(examples))] for i in range(0, len(examples), batch_size_mini)]
        
        processed_batches = []
        for idxes in batches_idx:
            data = {'nodes':torch.tensor(nodes_padded).long()[idxes[0]:idxes[1]], 'energy':torch.tensor(energy).float()[idxes[0]:idxes[1]], 'distance':torch.tensor(distance_padded).float()[idxes[0]:idxes[1]], 'node_extra':torch.tensor(nodes_extra_padded).float()[idxes[0]:idxes[1]]}
            processed_batches.append(data)


        return processed_batches



def query_database(mysql_url, idx_list):
    while True:
        try:
            db = connect(mysql_url)
            con = db._connect().cursor()

            
            input = {}
            cmps = [('id', ' IN ', '1,2')]

            keys = []  # No additional keys are needed for a simple ID query
            sort = None  # Assuming no sorting is required
            order = None  # Default order, can be 'DESC' for descending
            sort_table = None  # Not needed for an ID query
            columns = 'all'  # If you want all columns, otherwise specify the columns you need

            values = np.array([None for i in range(27)])
            values[25] = '{}'
            columnindex = list(range(27))

            what = ', '.join('systems.' + name
                                for name in
                                np.array(db.columnnames)[np.array(columnindex)])


            sql, args = db.create_select_statement(keys, cmps, sort, order,
                                                        sort_table, what)

            args = [tuple(idx_list)]


            con.execute(sql, args)

            deblob = db.deblob
            decode = db.decode

            rows = []

            for shortvalues in con.fetchall():
                values[columnindex] = shortvalues
                dct = {'id': values[0],
                            'unique_id': values[1],
                            'ctime': values[2],
                            'mtime': values[3],
                            'user': values[4],
                            'numbers': deblob(values[5], np.int32),
                            'positions': deblob(values[6], shape=(-1, 3)),
                            'cell': deblob(values[7], shape=(3, 3))}

                # if values[8] is not None:
                #     dct['pbc'] = (values[8] & np.array([1, 2, 4])).astype(bool)
                # if values[9] is not None:
                #     dct['initial_magmoms'] = deblob(values[9])
                # if values[10] is not None:
                #     dct['initial_charges'] = deblob(values[10])
                # if values[11] is not None:
                #     dct['masses'] = deblob(values[11])
                # if values[12] is not None:
                #     dct['tags'] = deblob(values[12], np.int32)
                # if values[13] is not None:
                #     dct['momenta'] = deblob(values[13], shape=(-1, 3))
                # if values[14] is not None:
                #     dct['constraints'] = values[14]
                # if values[15] is not None:
                #     dct['calculator'] = values[15]
                # if values[16] is not None:
                #     dct['calculator_parameters'] = decode(values[16])
                # if values[17] is not None:
                #     dct['energy'] = values[17]
                # if values[18] is not None:
                #     dct['free_energy'] = values[18]
                # if values[19] is not None:
                #     dct['forces'] = deblob(values[19], shape=(-1, 3))
                # if values[20] is not None:
                #     dct['stress'] = deblob(values[20])
                # if values[21] is not None:
                #     dct['dipole'] = deblob(values[21])
                # if values[22] is not None:
                #     dct['magmoms'] = deblob(values[22])
                # if values[23] is not None:
                #     dct['magmom'] = values[23]
                # if values[24] is not None:
                #     dct['charges'] = deblob(values[24])
                # if values[25] != '{}':
                #     dct['key_value_pairs'] = decode(values[25])
                if len(values) >= 27 and values[26] != 'null':
                    dct['data'] = decode(values[26], lazy=True)
                
                # external_tab = db._get_external_table_names()
                # tables = {}
                # for tab in external_tab:
                #     row = self._read_external_table(tab, dct["id"])
                #     tables[tab] = row

                # dct.update(tables)
                rows.append(AtomsRow(dct))
            return rows
        except Exception as e:
            time.sleep(1)  # 等待一秒再重试



def setGraphFea(row, distance):

    neighbors = 11
    num_atoms = len(distance)
    if num_atoms == 1:
        return np.zeros((num_atoms, 3 * neighbors * neighbors))
    embedding = np.zeros((num_atoms, 3 * neighbors * neighbors))
    if len(distance) < neighbors + 1:
        neighbors = num_atoms - 1
    sorted_idx = np.argsort(distance, axis=1)
    idx_cut = sorted_idx[:, 1:neighbors+1]

    i_indices = np.repeat(np.arange(num_atoms), neighbors**2)
    j_indices = np.repeat(np.expand_dims(idx_cut, axis=2), neighbors, axis=2).flatten()
    k_indices = idx_cut[idx_cut].flatten()

    angle_indices = [(i, j, k) for i, j, k in zip(i_indices, j_indices, k_indices)]
    # Vectorized angle calculation (assuming get_angles can accept arrays of tuples)
    angles = row.toatoms().get_angles(angle_indices)
    cosines = np.cos(np.radians(angles))

    angle_embedding = np.array(cosines).reshape(num_atoms, neighbors * neighbors)
    edge_ij = distance[i_indices, j_indices].reshape(num_atoms, neighbors * neighbors)
    edge_jk = distance[j_indices, k_indices].reshape(num_atoms, neighbors * neighbors)


    # Populate the output array
    for n in range(neighbors**2):
        embedding[:, 3*n] = edge_ij[:, n]
        embedding[:, 3*n + 1] = edge_jk[:, n]
        embedding[:, 3*n + 2] = angle_embedding[:, n]

    # atom_nbr_fea = np.array([position[indices] for indices in idx_cut])
    # centre_coords = np.expand_dims(position, axis=1)
    # centre_coords_expanded = np.repeat(centre_coords, repeats=neighbors, axis=1)
    # dxyz = atom_nbr_fea - centre_coords_expanded
    # r_cut = np.array([distance[i, idx_cut[i]] for i in range(0,len(distance))])
    # r = np.expand_dims(r_cut, axis=2)
    # angle_cosines = np.matmul(dxyz, np.swapaxes(dxyz, 1, 2)) / np.matmul(r, np.swapaxes(r, 1, 2))
    # embedding = 0



    return embedding





def GBF_distance_encode(matrix, min, max, step):

    gamma = (max - min) / (step - 1)
    filters = np.linspace(min, max, step)
    matrix = matrix[:, :, np.newaxis]
    matrix = np.tile(matrix, (1, 1, step))
    matrix = np.exp(-((matrix - filters) ** 2) / gamma**2)
    
    return matrix




def mask_elements_over_threshold(matrix, threshold):
    # 将超过阈值的元素替换为0
    masked_matrix = np.where(matrix > threshold, 0.0, matrix)
    return masked_matrix




def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr






def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metic_score(datas=np.array([ 5.9138,  0.    ,  1.1654,  0.    ,  7.3021,  0.    ,  1.1654, 0.    , 10.921 ])):
    """
    input datas: 1D array -- 9 elements -- this version
    only totl elec accepeted
    """
    eigenvalues, eigenvectors = np.linalg.eig(datas.reshape(3,3))
    D = np.diag(eigenvalues)
    return np.mean(np.diag(D))



def schmidt_orthogonalization(vectors):
    vectors = vectors.reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eig(vectors)
    diagonal_matrix = np.real(np.diag(eigenvalues))
    ele = np.mean(np.real(eigenvalues))

    return ele




@torch.no_grad()
def sample_from_model(model, x, points=None,test=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # device = torch.cuda.current_device()
    device = 'cpu'
    x = x.to(device)
    points = points.to(device)


    # print("x",x)
    # print("points",points)
    model = model.to(device)
    model.eval()
    N = len(x)
    _, predicts, _ = model(points, points=x)
    predicts = predicts.cpu()
    topN = 5
    targets = torch.tensor(np.arange(N))
    _, indx = predicts.topk(topN, 0)
    indx = indx.t()
    # correct = indx.eq(targets.unsqueeze(1).expand_as(indx))
    # correct_k = correct[:, :topN].view(-1).float().sum(0)
    # acc = correct_k.cpu().numpy() / N * 100
    indx = indx.cpu().numpy().tolist()
    vals = []
    for idxs in indx:  # 这个indx是距离最近的top5的那五个的索引
        val = [schmidt_orthogonalization(x[i, :].cpu().numpy()) for i in idxs]
        vals.append(val)
    vals = np.array(vals)
    print(vals)
    # 将数组转换为DataFrame对象
    # df = pd.DataFrame(vals)
    # # 保存DataFrame为CSV文件
    # csv_filename = 'pred.csv'
    # sucid = pd.read_csv("../success_id.csv", header=None)
    # df["ids"] = sucid
    # df.to_csv(csv_filename, index=False)
    # print('CSV 文件已保存为', csv_filename)
    if test:
        accuracy_sum = 0
        targets = targets.cpu().numpy()
        judge = torch.nn.MSELoss(reduce=True, size_average=True)
        error_diag = []
        error_diag_mae = []
        ground_truth = []
        best_pred = []
        for i in range(N):
            top3_captions = [indx[i][j] for j in range(len(indx[i]))]
            err = [judge(x[i, :], x[j, :]).cpu().numpy() for j in top3_captions]
            err_diag = [(schmidt_orthogonalization(x[i, :].cpu().numpy()) - schmidt_orthogonalization(x[j, :].cpu().numpy())) ** 2 for j in top3_captions]
            err_diag_mae = [
                abs(schmidt_orthogonalization(x[i, :].cpu().numpy()) - schmidt_orthogonalization(x[j, :].cpu().numpy()))
                for j in top3_captions]
            error_diag.append(np.min(err_diag))
            error_diag_mae.append(np.min(err_diag_mae))
            ground_truth.append(schmidt_orthogonalization(x[i, :].cpu().numpy()))
            best_pred.append(schmidt_orthogonalization(x[top3_captions[np.argmin(err_diag)], :].cpu().numpy()))
            if np.min(err) <= 1.5:
                accuracy_sum += 1

        best_pred = np.array(best_pred)
        ground_truth = np.array(ground_truth)
        rss = np.sum((ground_truth - best_pred) ** 2)

        # 计算总平方和
        tss = np.sum((ground_truth - np.mean(ground_truth)) ** 2)

        # 计算R2分数
        r2 = 1 - (rss / tss)
        #
        import matplotlib.pyplot as plt

        plt.scatter(ground_truth, best_pred, s=10)
        plt.plot([min(ground_truth), max(ground_truth)],
                [min(ground_truth), max(ground_truth)], 'r--', label='y=x')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Scatter Plot of Predicted Values')
        plt.legend()
        plt.savefig('test_set.svg', format='svg')
        plt.show()
        return r2



class CharDataset(Dataset):
    def __init__(self, node_data, edge_data, graph_data, response, max_length, node_embd_len, graph_embd_len):

        self.data_size = len(node_data)
        print('data has %d examples' % (self.data_size))

        self.node_data = node_data
        self.edge_data = edge_data
        self.graph_data = graph_data
        self.response = response

        self.max_length = max_length
        self.node_embd_len = node_embd_len
        self.graph_embd_len = graph_embd_len

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        
        node = np.array(self.node_data[idx])
        n_atoms = len(node)

        node_padded = torch.zeros(self.max_length, self.node_embd_len)
        node_padded[:n_atoms, :] = torch.tensor(node).float()

        edge = np.array(self.edge_data[idx])


        edge_rbf_bins = 12
        edge_extr = edge[:, :, np.newaxis]
        edge_extr = np.repeat(edge_extr, edge_rbf_bins, axis=-1)


        edge = mask_elements_over_threshold(1.0 / edge, 8.0)
        edge_zero_extr = edge[:, :, np.newaxis]
        edge_zero_extr = np.repeat(edge_zero_extr, edge_rbf_bins, axis=-1)

        edge_embedding = GBF_distance_encode(edge, 0.0, 8.0, edge_rbf_bins)
        edge_embedding[edge_extr == 1] = 1.0
        edge_embedding[edge_zero_extr == 0] = 0.0
        edge_padded = torch.zeros(self.max_length, self.max_length, edge_rbf_bins)
        edge_padded[:n_atoms, :n_atoms, :] = torch.tensor(edge_embedding).float()
        


        graph = torch.tensor(np.array(self.graph_data[idx])).float()

        mask = torch.zeros(self.max_length).long()
        mask[:n_atoms] = 1

        energy = torch.tensor(np.array(self.response[idx])).float()

        return node_padded, edge_padded, graph, mask, energy

def processDataFiles(files):
    text = ""
    for f in tqdm(files):
        with open(f, 'r') as h:
            lines = h.read() # don't worry we won't run out of file handles
            if lines[-1]==-1:
                lines = lines[:-1]
            #text += lines #json.loads(line)
            text = ''.join([lines,text])
    return text

