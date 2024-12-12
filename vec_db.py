from typing import Dict, List, Annotated
import numpy as np
import os
from pq import PQ
from ivf import IVF
import struct
import pickle

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.db_size = db_size
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index(rows)

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"
        
    def get_cluster_vectors(self, ids):
        cluster_vectors = []
        with open(self.db_path, "rb") as file:
            for id in ids:
                file.seek(id * DIMENSION * ELEMENT_SIZE)
                data = file.read(DIMENSION * ELEMENT_SIZE)
                vector = struct.unpack(f'{DIMENSION}f', data)
                cluster_vectors.append(vector)
            file.close()
        return np.array(cluster_vectors)
    
    def get_cluster_vectors_one_by_one(self, ids):
        cluster_vectors = []
        for id in ids:
            vector = self.get_one_row(id)
            cluster_vectors.append(vector)
        return np.array(cluster_vectors)


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        ivf = IVF()
        print("reading centroids")
        with open(f"./{self.index_path}/centroids.pkl", "rb") as f:
            centroids = pickle.load(f)
            f.close()
            del f
        print("done reading centroids")
        MAX_CLUSTER_SIZE = self.db_size // len(centroids)
        return ivf.search(query, self.index_path, centroids, self, top_k, max_loaded_clusters=(top_k + (top_k * DIMENSION * 16) // MAX_CLUSTER_SIZE))
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, data):
        # Placeholder for index building logic
        # pq = PQ()
        # pq.generate_code_books(data)
        # pq.encode_data(data)
        # return
        of_clusters = 15624
        batch_size = of_clusters // 10

        ivf = IVF()
        ivf.train(data, self.index_path, of_clusters, batch_size)
        pass

        


