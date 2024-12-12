from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle
from datetime import datetime
from memory_profiler import profile
import os
from gpu_kmeans import GPUKMeans

##############################################################################################################################################
############################################################ INVERTED FILE INDEX #############################################################
##############################################################################################################################################

class IVF:

    no_of_clusters = 20000
    batch_size     = 10000

    def train(self, training_data, index_path, no_of_clusters=20000, batch_size=10000):
        self.no_of_clusters = no_of_clusters
        self.batch_size = batch_size
        print("Clustering Kmeans...")
        start_time = datetime.now()

        print("Normalizing input data...")
        data_copy = training_data.copy()
        norms = np.linalg.norm(data_copy, axis=1, keepdims=True)
        data_copy = np.divide(data_copy, norms, where=norms != 0)

        # Initialize and fit GPU KMeans
        kmeans = GPUKMeans(n_clusters=self.no_of_clusters)
        kmeans.fit(data_copy, batch_size=self.batch_size)
        print(f"\nTotal clustering time: {datetime.now() - start_time}")

        assignments = kmeans.labels_
        cluster_to_ids: dict[int: list[int]] = {}
        os.makedirs(f"./{index_path}/clusters", exist_ok=True)

        for i in range(0, self.no_of_clusters):
            cluster_to_ids[i] = []

        for i in range(0, len(assignments)):
            cluster_to_ids[assignments[i]].append(i)

        for cluster, ids in cluster_to_ids.items():
            with open(f"./{index_path}/clusters/cluster_{cluster}.pkl", "wb") as f:
                pickle.dump(ids, f)
                f.close()
        with open(f"./{index_path}/centroids.pkl", "wb") as f:
            pickle.dump(kmeans.cluster_centers_, f)
            f.close()
        print("Clustering successful")
        return cluster_to_ids
    
    def cos_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def search(self, query, index_path, centroids, db, k: int, max_loaded_clusters=5):
        query = np.array(query).reshape((70,))
        closest_clusters = np.argsort([(self.cos_similarity(query.T, centroid)) for centroid in centroids]).tolist()[::-1]
        similarities = []
        loader_clusters = 0
        for closest_cluster in closest_clusters:
            loader_clusters += 1
            with open(f"./{index_path}/clusters/cluster_{closest_cluster}.pkl", "rb") as f:
                ids = pickle.load(f)
                f.close()
                del f
            print("fetching cluster...")
            vectors = db.get_cluster_vectors_one_by_one(ids)
            print("cluster fetched")
            i = 0
            for vector in vectors:
                cos = self.cos_similarity(query, vector)
                similarities.append((cos, ids[i]))
                i += 1
            del vectors
            if loader_clusters >= max_loaded_clusters and len(similarities) >= k:
                break
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        return [d[1] for d in similarities[:k]]