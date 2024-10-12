import argparse  # Import the argparse module
from Crylearn import cry2graph 
from ase.db import connect
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_entry(k, database_path):
    """
    Processes an entry in the database to extract information for further use.

    Parameters:
    k (int): Index of the entry in the database.
    database_path (str): Path to the database.

    Returns:
    tuple: A tuple containing atoms, formation energy, node embedding, adjacency matrix, and global info if successful.
           Returns None in case of an error.
    """
    _id = k + 1
    try:
        database = connect(database_path)
        item = database.get(id=_id)
        atoms = item.toatoms()
        node_embedding, adj_matrix, dis_matrix, global_info = cry2graph.parser(database, _id).get()
        
        if item['band_gap'] < 1e-15:
            metal = 1
        else : metal = 0 # non-metal

        response = {
                    'formation_energy': item['formation_energy'],
                    'band_gap': item['band_gap'],
                    'bulk_modulus': item['bulk_modulus'],
                    'metal' : metal
                }
        
        return atoms, response, node_embedding, adj_matrix, global_info
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
    # Connect to the source database and get the total number of entries
    database = connect(database_path)
    total_entries = database.count()
    
    # Connect to the new database for writing filtered results
    newdb = connect(new_database_path)

    # Use ProcessPoolExecutor to process entries in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_entry, k, database_path): k for k in range(total_entries)}
        
        # Process results as they are completed
        for future in tqdm(as_completed(futures), total=total_entries, desc='Writing to Database'):
            result = future.result()
            
            if result is not None:
                atoms, response, node_embedding, adj_matrix, global_info = result
                
                # Filter out entries based on node embedding size
                if len(node_embedding) > 500:
                    continue
                
                # Write filtered data to the new database
                data = {
                    'node_embedding': node_embedding,
                    'adj_matrix': adj_matrix,
                    'global_info': global_info
                }
                newdb.write(atoms=atoms,formation_energy=response['formation_energy'], 
                            band_gap=response['band_gap'],bulk_modulus=response['bulk_modulus'], 
                            metal=response['metal'], data=data)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filter and write crystal structures from one database to another.")
    parser.add_argument(
        '--database_path', 
        type=str, 
        required=True, 
        help='Path to the source database containing crystal structures.'
    )
    parser.add_argument(
        '--new_database_path', 
        type=str, 
        required=True, 
        help='Path to the destination database to store filtered results.'
    )

    args = parser.parse_args()  # Parse the arguments

    # Start the filtering and writing process
    filter_and_write_database(args.database_path, args.new_database_path, process_entry)
