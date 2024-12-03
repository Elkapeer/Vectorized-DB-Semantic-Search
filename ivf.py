from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle
from datetime import datetime
from memory_profiler import profile
import os
##############################################################################################################################################
############################################################ INVERTED FILE INDEX #############################################################
##############################################################################################################################################

class IVF:

    no_of_clusters = 100

    def train(self, training_data):
        print("Clustering Kmeans...")
        start_time = datetime.now()
        kmeans = MiniBatchKMeans(n_clusters=self.no_of_clusters, batch_size=10000)
        kmeans.fit(training_data)
        print(f"Finished in: {datetime.now() - start_time}")
        assignments = kmeans.labels_
        cluster_to_ids : dict[int: list[int]]= {}
        for i in range(0, self.no_of_clusters):
            cluster_to_ids[i] = []
        for i in range(0, len(assignments)):
            cluster_to_ids[assignments[i]].append(i)
        for cluster, ids in cluster_to_ids.items():
            with open(os.path.join("clusters", f"cluster_{cluster}.pkl"), "wb") as f:
                pickle.dump(ids, f)
                f.flush()
        with open("centroids.pkl", "wb") as f:
            pickle.dump(kmeans.cluster_centers_, f)
            f.flush()
        print("Clustering successful")
        return cluster_to_ids
    
    def cos_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def search(self, query, centroids, db, k : int):
        query = np.array(query).reshape((1,-1))
        closest_clusters = np.argsort([np.linalg.norm(centroid - query) ** 2 for centroid in centroids])[:k] # TODO: return more closest clusters
        similarities = []
        for closest_cluster in closest_clusters:
            with open(os.path.join("clusters", f"cluster_{closest_cluster}.pkl"), "rb") as f:
                ids = pickle.load(f)
                f.close()
                del f
            progress = 0
            total_progress = len(ids)
            print("fetching cluster...")
            vectors = db.get_cluster_vectors(ids)
            print("cluster fetched")
            i = 0
            for vector in vectors:
                cos = self.cos_similarity(query, vector)
                similarities.append((cos, ids[i]))
                i += 1
                #progress += 1
                #print(f"{progress}/{total_progress}")
            del vectors
        similarities = sorted(similarities, key= lambda x : x[0], reverse=True)
        return [d[1] for d in similarities[:k]]