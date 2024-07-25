# Author: Bin CAO <barniecao@outlook.com>

from Crylearn import cry2graph
from ase.db import connect
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_entry(k, database_path):
    _id = k + 1
    try:
        database = connect(database_path)
        item = database.get(id=_id)
        atoms = item.toatoms()
        node_embedding, adj_matrix, dis_matrix, global_info = cry2graph.parser(database, _id).get()
        formation_energy = item['formation_energy']
        return atoms, formation_energy, node_embedding, adj_matrix, global_info
    except Exception as e:
        print(f"An error occurred for ID {_id}: {e}")
        return None

def filter_and_write_database(database_path, new_database_path, process_entry):
    """
    Filters and writes data from one database to another using parallel processing.

    Parameters:
    database_path (str): Path to the source database.
    new_database_path (str): Path to the destination database.
    process_entry (function): Function to process each database entry.
    """
    # Connect to the source database
    database = connect(database_path)
    total_entries = database.count()
    
    # Connect to the new database
    newdb = connect(new_database_path)

    # Use ProcessPoolExecutor to process entries in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_entry, k, database_path): k for k in range(total_entries)}
        for future in tqdm(as_completed(futures), total=total_entries, desc='Writing to Database'):
            result = future.result()
            if result is not None:
                atoms, formation_energy, node_embedding, adj_matrix, global_info = result
                if len(node_embedding) > 300:
                    continue
                data = {
                    'node_embedding': node_embedding,
                    'adj_matrix': adj_matrix,
                    'global_info': global_info
                }
                newdb.write(atoms=atoms, formation_energy=formation_energy, data=data)



if __name__ == "__main__":
    database_path = '/home/cb/cb_crystal/test_ASUnet/temp/structures.db'
    new_database_path = './temp/filter_self_struc_cif.db'

    filter_and_write_database(database_path, new_database_path, process_entry)                                       
