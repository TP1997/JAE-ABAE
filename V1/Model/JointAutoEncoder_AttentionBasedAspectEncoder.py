# Use pytorch_py3.8.8

import numpy as np

import torch 
from torch.nn.parameter import Parameter
from torch.nn import init

class SelfAttention(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.M = Parameter(torch.empty(size=(emb_dim, emb_dim)))
        init.kaiming_uniform_(self.M.data)
        
        self.attention_softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, e):
        y = torch.mean(e, (1,)).unsqueeze(2)
        d_temp = torch.matmul(self.M, y)
        d = torch.matmul(e, d_temp).squeeze(2)
        attention_weights = self.attention_softmax(d)
        
        return attention_weights
 
class JAE_ABAE(torch.nn.Module):
    def __init__(self, args, word_embeddings, 
                 init_topic_matrix_p1=None, init_topic_matrix_p2=None, init_topic_matrix_s12=None):
        super().__init__()
        
        # Member variables
        self.emb_dim = args['emb_dim']
        self.n_topics_private1 = args['n_topics_private1']
        self.n_topics_shared = args['n_topics_shared']
        self.n_topics_private2 = args['n_topics_private2']
        self.ortho_reg = args['ortho_reg']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        # Create Identity matrices for every topic matrix size.
        self.I_p1 = Parameter(torch.eye(self.n_topics_private1), requires_grad=False)
        self.I_s12 = Parameter(torch.eye(self.n_topics_shared), requires_grad=False)
        self.I_p2 = Parameter(torch.eye(self.n_topics_private2), requires_grad=False)
        
        # Embedding layer to transform sequence of indices to sequence of vectors.
        self.word_embedding = torch.nn.Embedding.from_pretrained(word_embeddings,
                                                                  freeze=True)
        # Branch-specific attention mechanisms.
        self.attention_p1 = SelfAttention(self.emb_dim)
        self.attention_s1 = SelfAttention(self.emb_dim)
        self.attention_p2 = SelfAttention(self.emb_dim)
        self.attention_s2 = SelfAttention(self.emb_dim)
        
        # Branch-specific linear transformations.
        self.linear_transform_p1 = torch.nn.Linear(self.emb_dim, self.n_topics_private1)
        self.linear_transform_s1 = torch.nn.Linear(self.emb_dim, self.n_topics_shared)
        self.linear_transform_p2 = torch.nn.Linear(self.emb_dim, self.n_topics_private2)
        self.linear_transform_s2 = torch.nn.Linear(self.emb_dim, self.n_topics_shared)
        self.softmax_topics = torch.nn.Softmax(dim=-1)
        
        # Branch-specific topic embedding matrices.
        self.topic_embeddings_p1 = Parameter(torch.empty(size=(self.emb_dim, self.n_topics_private1)))
        self.topic_embeddings_p2 = Parameter(torch.empty(size=(self.emb_dim, self.n_topics_private2)))
        self.topic_embeddings_s12 = Parameter(torch.empty(size=(self.emb_dim, self.n_topics_shared)))
        
        # Initialize topic embeddings.
        if init_topic_matrix_p1 is None:
            torch.nn.init.xavier_uniform(self.topic_embeddings_p1)
            torch.nn.init.xavier_uniform(self.topic_embeddings_p2)
            torch.nn.init.xavier_uniform(self.topic_embeddings_s12)
        else:
            self.topic_embeddings_p1.data = torch.from_numpy(init_topic_matrix_p1.T)
            self.topic_embeddings_p2.data = torch.from_numpy(init_topic_matrix_p2.T)
            self.topic_embeddings_s12.data = torch.from_numpy(init_topic_matrix_s12.T)
            
    def document_topic_dist(self, text_embeddings, branch: str):
        # Encoder.
        # Input: Word embeddings of a single sentence.
        # Output: Attention weights, document-topic distribution, document embedding.
        
        # Calculate branch-specific attention weights and sentence embeddings
        attention_weights = None
        doc_emb = None
        doc_emb_reduced = None
        if branch=='p1':
            attention_weights = self.attention_p1(text_embeddings)
            doc_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
            doc_emb_reduced = self.linear_transform_p1(doc_emb)
        elif branch=='s1':
            attention_weights = self.attention_s1(text_embeddings)
            doc_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
            doc_emb_reduced = self.linear_transform_s1(doc_emb)
        elif branch=='p2':
            attention_weights = self.attention_p2(text_embeddings)
            doc_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
            doc_emb_reduced = self.linear_transform_p2(doc_emb)
        elif branch=='s2':
            attention_weights = self.attention_s2(text_embeddings)
            doc_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
            doc_emb_reduced = self.linear_transform_s2(doc_emb)
            
        # Document-topic distribution, Equation (6)
        p = self.softmax_topics(doc_emb_reduced)
        
        return attention_weights, p, doc_emb
        
    @staticmethod
    def _reconstruction_loss(doc_emb, doc_emb_rec, neg_emb):
        # Unregularized loss, Equation (7)
        rz = torch.matmul(doc_emb.unsqueeze(1), doc_emb_rec.unsqueeze(2)).squeeze()
        rn = torch.matmul(neg_emb, doc_emb_rec.unsqueeze(2)).squeeze()
        loss = torch.sum(1 - rz.unsqueeze(1) + rn, dim=1)
        
        max_margin = torch.max(loss, torch.zeros_like(loss)).unsqueeze(dim=-1)
        
        return max_margin
        
    def orthogonality_regularization(self):
        # Calculate regularization term for each topic matrix, Equation (8).
        reg_loss = torch.norm(torch.matmul(self.topic_embeddings_p1.t(), self.topic_embeddings_p1) - self.I_p1) +\
                    torch.norm(torch.matmul(self.topic_embeddings_p2.t(), self.topic_embeddings_p2) - self.I_p2) +\
                    torch.norm(torch.matmul(self.topic_embeddings_s12.t(), self.topic_embeddings_s12) - self.I_s12)
        return reg_loss
    
    def forward(self, text_embeddings_idx_r1, text_embeddings_idx_r2, negative_samples_idx_r1, negative_samples_idx_r2):
        
        # Transform indexes to embeddings.
        text_embeddings_r1 = self.word_embedding(text_embeddings_idx_r1)
        text_embeddings_r2 = self.word_embedding(text_embeddings_idx_r2)
        negative_samples_r1 = self.word_embedding(negative_samples_idx_r1)
        negative_samples_r2 = self.word_embedding(negative_samples_idx_r2)
            
        # Average negative samples
        negative_samples_mean_r1 = torch.mean(negative_samples_r1, dim=2)
        negative_samples_mean_r2 = torch.mean(negative_samples_r2, dim=2)
        
        # Encoding: Get branch-specific topic importances and document embedding, Equations (1) & (6)
        _, p_p1, doc_emb_p1 = self.document_topic_dist(text_embeddings_r1, branch='p1')
        _, p_s1, doc_emb_s1 = self.document_topic_dist(text_embeddings_r1, branch='s1')
        _, p_p2, doc_emb_p2 = self.document_topic_dist(text_embeddings_r2, branch='p2')
        _, p_s2, doc_emb_s2 = self.document_topic_dist(text_embeddings_r2, branch='s2')
        
        # Calculate convex combinations to represent the original document: (1-λ_i)*doc_emb_pi + λ_*doc_emb_si.
        doc_emb_1 = (1-self.lambda1)*doc_emb_p1 + self.lambda1*doc_emb_s1
        doc_emb_2 = (1-self.lambda2)*doc_emb_p2 + self.lambda2*doc_emb_s2
        
        # Decoding: Get branch-specific document embedding reconstructions, Equation (5)
        doc_emb_reconstruction_p1 = torch.matmul(self.topic_embeddings_p1, p_p1.unsqueeze(2)).squeeze()
        doc_emb_reconstruction_s1 = torch.matmul(self.topic_embeddings_s12, p_s1.unsqueeze(2)).squeeze()
        doc_emb_reconstruction_p2 = torch.matmul(self.topic_embeddings_p2, p_p2.unsqueeze(2)).squeeze()
        doc_emb_reconstruction_s2 = torch.matmul(self.topic_embeddings_s12, p_s2.unsqueeze(2)).squeeze()
            
        # Calculate convex combinations for outputs of p1&s1 and p2&s2
        doc_emb_recon_1 = (1-self.lambda1)*doc_emb_reconstruction_p1 + self.lambda1*doc_emb_reconstruction_s1
        doc_emb_recon_2 = (1-self.lambda2)*doc_emb_reconstruction_p2 + self.lambda2*doc_emb_reconstruction_s2
        
        # Calculate unregularized reconstruction loss.
        recon_loss_1 = JAE_ABAE._reconstruction_loss(doc_emb_1, doc_emb_recon_1, negative_samples_mean_r1)
        recon_loss_2 = JAE_ABAE._reconstruction_loss(doc_emb_2, doc_emb_recon_2, negative_samples_mean_r2)
            
        return recon_loss_1 + recon_loss_2 + self.ortho_reg * self.orthogonality_regularization()
        
    def get_topic_words(self, topics_type, wv_model, topn=15):
        # Select which topic to obtain.
        topic_emb = None
        if topics_type == 'private_1':
            topic_emb = self.topic_embeddings_p1.detach().cpu().numpy()
        elif topics_type == 'private_2':
            topic_emb = self.topic_embeddings_p2.detach().cpu().numpy()
        elif topics_type =='shared':
            topic_emb = self.topic_embeddings_s12.detach().cpu().numpy()
            
        words = []
        # Cosine distance between words and topics.
        words_scores = wv_model.wv.vectors.dot(topic_emb)
            
        for row in range(topic_emb.shape[1]):
            argmax_words_scores = np.argsort(-words_scores[:, row])[:topn]
            words.append([wv_model.wv.index_to_key[i] for i in argmax_words_scores])
                
        return words

            
            
            
            
            
            
            
            
            
            
            
            
            
            
        