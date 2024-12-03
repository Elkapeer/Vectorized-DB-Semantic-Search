from sklearn.cluster import KMeans
import numpy as np
import pickle

##############################################################################################################################################
############################################################ PRODUCT QUANTIZATION ############################################################
##############################################################################################################################################

class PQ:
    
    # vector dimension
    dimension = 70 

    # number of subvectors for each vector
    no_of_subspaces = 5
    assert dimension % no_of_subspaces == 0, "Dimension must be divisible by number of subspaces"

    # number of clusters for kmeans to build the codebook
    no_of_clusters = 64

    # Divide data to subspaces 
    def get_subspaces(self, data):
        data = np.array(data)
        if data.ndim == 1: # just 1 vector
            data = np.array(data).reshape(1, -1) # so i can use np.split() lol
        return np.array(np.split(data, self.no_of_subspaces, axis=1))
    
    # Generates the codebooks that maps each subvector to a centroid
    def generate_code_books(self, training_data):
        progress = 0
        codebooks = []
        subspaces = self.get_subspaces(training_data)
        print("Generating Codebook...")
        for subspace in subspaces:
            # Kmeans estimator to train on each supspace
            ###################################### TODO: may change this parameters ################################################
            kmeans = KMeans(n_clusters=self.no_of_clusters, random_state=42) 
            kmeans.fit(subspace)
            # Kmeans centers are our codebook for each supspace
            codebooks.append(kmeans.cluster_centers_)
            progress += 1
            print(f"Progress: {progress}/{len(subspaces)}")
        codebooks = np.array(codebooks)
        with open("pq_codebooks.pkl", "wb") as f:
            pickle.dump(codebooks, f)
        print("Codebook generated successfully")
        return codebooks
    
    # Encode our database vectors (convert every vector to its centroids)
    def encode_data(self, data):
        with open("pq_codebooks.pkl", "rb") as f:
            codebooks = pickle.load(f)
        subspaces = self.get_subspaces(data)
        encoded_data : list[list[int]] = []
        progress = 0
        total_progress = self.no_of_subspaces * len(subspaces[0])
        print("Encoding data...")
        for i in range(0, self.no_of_subspaces):
            encoded_supspace : list[int] = []
            # for each supspace
            for subvector in subspaces[i]:
                # find nearest centroid in the codebook for this supspace
                encoded_supspace.append(np.argmin([np.linalg.norm(subvector - codeword) for codeword in codebooks[i]]))
                progress += 1
                print(f"Progress: {progress}/{total_progress}")
            encoded_data.append(encoded_supspace)
        encoded_data = np.array(encoded_data).T
        with open("pq_data.pkl", "wb") as f:
            pickle.dump(encoded_data, f)
        print("Data compressed successfully")
        return encoded_data
    
    # Generates the distance look-up table to reduce computational time when calculating distance between vectors in searching
    # table has dimensions of (#clusters, #subspaces)
    def generate_distance_table(self, query, codebook):
        supspaces = self.get_subspaces(query)
        distance_table = np.zeros((self.no_of_clusters, self.no_of_subspaces))
        # for each cluster (centroid) calculate squared euclidean distance between the subspace and the codebook for this supspace
        for i in range(0, self.no_of_subspaces):
            diff = codebook[i] - supspaces[i] # difference between supspace and its codebook
            diff = diff ** 2
            # sum differences to get the squared euclidean distance
            supspace_distance = np.array(np.sum(diff, axis=1)) # dimensions: (#clusters, )
            distance_table[:, i] = supspace_distance
        return distance_table
    
    # Query a vector to get k closest vectors
    def search(encoded_data, distance_table, k):
        distances = []
        # for each encoded vector in our database
        for code in encoded_data: 
            distance = 0
            i = 0
            # for each centroid in this vector
            for centroid in code: 
                # get the distance to this centroid from the distance table and sum distances up
                distance += distance_table[centroid][i]
                i += 1
            distances.append(distance)

        # get the indices of k vectors with smallest distances with the query vector
        indices = np.argsort(distances)[:k]
        return indices