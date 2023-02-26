# Use pytorch_py3.8.8
import numpy as np
import json
from sklearn.cluster import MiniBatchKMeans
import gensim


def get_w2v(path):
    # Load a previously saved Word2Vec model.
    return gensim.models.Word2Vec.load(path)


def text2vocab_idx(text: list, vocabulary, pad_idx, maxlen):
    # Transform sentence into a list of vocabulary indices
    # pad_idx padding until maxlen
    vocab_idxs = [vocabulary[w] for w in text]
    
    if len(vocab_idxs) < maxlen:
        vocab_idxs.extend([pad_idx] * (maxlen - len(vocab_idxs)))
        
    return vocab_idxs
            

def read_data_batches_multifile(text_path1, text_path2, batch_size=50):
    # Reading batched texts
    batch1 = []
    batch2 = []
    
    with open(text_path1, 'r', encoding='utf-8') as f1, open(text_path2, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip().split()
            line2 = line2.strip().split()
            
            batch1.append(line1)
            batch2.append(line2)
            
            if len(batch1) >= batch_size:
                yield batch1, batch2
                batch1 = []
                batch2 = []
            

def read_data_indices_multifile(data_conf, pad_idx, batch_size=50):
    # Data generator for training the model
    # From text file to vocabulary index sequences batches
    
    # Vocabulary is based on word2vec model trained on R_1 ∪ R_2
    w2v_model = get_w2v(data_conf['wv_path'])
    vocabulary = w2v_model.wv.key_to_index
     
    
    for batch_R1, batch_R2 in read_data_batches_multifile(data_conf['R1']['text_path'], data_conf['R2']['text_path'], batch_size):
        batch_index_vecs_R1 = []
        batch_index_vecs_R2 = []
        
        for text_R1, text_R2 in zip(batch_R1, batch_R2):
            # Transform texts into a list of vocabulary indices.
            text_R1_vocab_idx = text2vocab_idx(text_R1, vocabulary, pad_idx, data_conf['R1']['maxlen'])
            text_R2_vocab_idx = text2vocab_idx(text_R2, vocabulary, pad_idx, data_conf['R2']['maxlen'])
            
            batch_index_vecs_R1.append(np.asarray(text_R1_vocab_idx, dtype=np.int64))
            batch_index_vecs_R2.append(np.asarray(text_R2_vocab_idx, dtype=np.int64))
    
        # Output dimension: (batch_size, maxlen)
        yield np.stack(batch_index_vecs_R1, axis=0), np.stack(batch_index_vecs_R2, axis=0)
        
        

def get_centroids(data_matrix, k, normalize=True):
    km = MiniBatchKMeans(n_clusters=k, n_init=100)
    km.fit(data_matrix)
    clusters = km.cluster_centers_
    
    # L2 normalization
    if normalize:
        clusters = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        
    return clusters

        
def init_topic_matrices(path, path1, path2, jae_abae_conf):
    w2v_model1 = get_w2v(path1)
    w2v_model2 = get_w2v(path2)
    w2v_model_jae_abae = get_w2v(path)
    
    # Obtain words used in R1\{R1 ∩ R2}, R2\{R1 ∩ R2} and R1 ∩ R2.
    R1_words = set(w2v_model1.wv.key_to_index.keys())
    R2_words = set(w2v_model2.wv.key_to_index.keys())
    R1andR2_words = R1_words.intersection(R2_words)
    R1_words = R1_words.difference(R1andR2_words)
    R2_words = R2_words.difference(R1andR2_words)
    
    # Obtain word indices of above words wrt. w2v_model_jae_abae.
    R1_idx = [ w2v_model_jae_abae.wv.key_to_index[w] for w in R1_words]
    R2_idx = [ w2v_model_jae_abae.wv.key_to_index[w] for w in R2_words]
    R1andR2_idx = [ w2v_model_jae_abae.wv.key_to_index[w] for w in R1andR2_words]
    
    # Obtain K-means centroids.
    T1_p_init = get_centroids(w2v_model_jae_abae.wv.vectors[R1_idx], k=jae_abae_conf['n_topics_private1'])
    T2_p_init = get_centroids(w2v_model_jae_abae.wv.vectors[R2_idx], k=jae_abae_conf['n_topics_private2'])
    T12_s_init = get_centroids(w2v_model_jae_abae.wv.vectors[R1andR2_idx], k=jae_abae_conf['n_topics_shared'])
    
    return T1_p_init, T2_p_init, T12_s_init


def save_topics(topics: list, savepath):
    with open(savepath, 'w') as f:
        for i, t in enumerate(topics):
            words = ' '.join([w for w in t])
            f.write(f'Topic {i+1}:\n')
            f.write(words+'\n\n')


def save_topics_json(topics: list, savepath):
    topic_dict = { i:t for i,t in enumerate(topics)}
    
    with open(savepath, 'w', encoding='utf-8') as f:
        json.dump(topic_dict, f, ensure_ascii=False, indent=4)
            












# Old code:
def read_data_batches(text_path, batch_size=50):
    # Reading batched texts
    batch = []
    
    for line in open(text_path, encoding='utf-8'):
        line = line.strip().split()
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
            
    # Causes strange behaviour in loss.
    #if len(batch) > 0:
        #yield batch
        

def read_data_indices(text_path, vocabulary: dict, pad_idx, batch_size=50, maxlen=100):
    # Data for training the model
    # From text file to vocabulary index sequences batches
        
    for batch in read_data_batches(text_path, batch_size):
        batch_index_vecs = []
        
        for text in batch:
            # Transform sentence into a list of vocabulary indices
            text_as_vocab_idxs = text2vocab_idx(text, vocabulary, pad_idx, maxlen)
            batch_index_vecs.append(np.asarray(text_as_vocab_idxs, dtype=np.int64))
            
        # Output dimension: (batch_size, maxlen)
        yield np.stack(batch_index_vecs, axis=0)
        
        
def get_centroids2(w2v_model, k, normalize=True):
    # Clustering all word vectors with K-means and returning L2-normalizes
    # cluster centroids; used for ABAE aspects matrix initialization.
    
    km = MiniBatchKMeans(n_clusters=k, n_init=100)
    m = []
    
    for k in w2v_model.wv.key_to_index:
        m.append(w2v_model.wv[k])
        
    m = np.matrix(m)
    km.fit(m)
    clusters = km.cluster_centers_
    
    # L2 normalization
    # Why normalization?
    if normalize:
        clusters = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
    
    return clusters #norm_topic_matrix_init

