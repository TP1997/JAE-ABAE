# Use pytorch_py3.8.8
import sys
sys.path.insert(1, '/home/tuomas/Python/Gradu/Pytorch code/JAE-ABAE_model_pytorch/Model') 

from JointAutoEncoder_AttentionBasedAspectEncoder import JAE_ABAE
from reader import init_topic_matrices, get_w2v, read_data_indices_multifile, save_topics

import torch
import logging
import json
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    # filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load dataset configuration.
data_name = ['All_Beauty',
             'AMAZON_FASHION',
             'CDs_and_Vinyl',
             'Cell_Phones_and_Accessories',
             'Digital_Music',
             'Electronics',
             'Industrial_and_Scientific',
             'Luxury_Beauty',
             'Musical_Instruments',
             'Software',
             'Video_Games']
d1 = 9
d2 = 1
subdir='200k/'
data_dir = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/JAE_ABAE/{data_name[d1]}&{data_name[d2]}/{subdir}'

with open(data_dir + 'data_conf.json') as f:
    data_conf = json.load(f)
    
#%% Load word embeddings.

w2v_model = get_w2v(data_conf['wv_path'])

#%% Define JAE-ABAE-model

# JAE-ABAE configuration.
abae_dir = "/home/tuomas/Python/Gradu/Pytorch code/JAE-ABAE_model_pytorch/Model/"
with open(abae_dir + 'JAE-ABAE_conf.json') as f:
    jae_abae_conf = json.load(f)
    
# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the model
T1_p_init, T2_p_init, T12_s_init = init_topic_matrices(data_conf['wv_path'],
                                                       data_conf['R1']['wv_path'],
                                                       data_conf['R2']['wv_path'],
                                                       jae_abae_conf)
model = JAE_ABAE(jae_abae_conf,
                 torch.from_numpy(np.vstack([w2v_model.wv.vectors, np.zeros(jae_abae_conf['emb_dim'], dtype='float32')])),     # Word embedding vectors + zero vector for padding index.
                 init_topic_matrix_p1=T1_p_init,
                 init_topic_matrix_p2=T2_p_init,
                 init_topic_matrix_s12=T12_s_init
                 )
model = model.to(device)

if jae_abae_conf['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif jae_abae_conf['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters())
elif jae_abae_conf['optimizer'] == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())
elif jae_abae_conf['optimizer'] == 'asgd':
    optimizer = torch.optim.ASGD(model.parameters())
else:
    raise Exception('Optimizer {} is not supported.'.format(jae_abae_conf['optimizer']))
    
#%% Train JAE-ABAE-model
    
criterion = torch.nn.MSELoss(reduction='mean')
criterion_target = torch.zeros((jae_abae_conf['batch_size'], 1))
criterion_target = criterion_target.to(device)
    
log_topics = False
pad_idx = w2v_model.wv.vectors.shape[0]
maxsize = max(data_conf['R1']['size'], data_conf['R2']['size'])

for e in range(jae_abae_conf['epochs']):

    data_iterator = read_data_indices_multifile(data_conf, pad_idx, jae_abae_conf['batch_size'])
    
    with tqdm(data_iterator, total=int(maxsize/jae_abae_conf['batch_size'])) as tqdb_batch_iter:
        tqdb_batch_iter.set_description(f"Epoch {e+1}/{jae_abae_conf['epochs']}")
        for batch_R1, batch_R2 in tqdb_batch_iter:
            
            # Pad with pad_idx if smaller than batch size
            if batch_R1.shape[0] < jae_abae_conf['batch_size']:
                batch_R1 = np.pad(batch_R1, ((0, jae_abae_conf['batch_size'] - batch_R1.shape[0]), (0,0)), constant_values=pad_idx)
            if batch_R2.shape[0] < jae_abae_conf['batch_size']:
                batch_R2 = np.pad(batch_R2, ((0, jae_abae_conf['batch_size'] - batch_R2.shape[0]), (0,0)), constant_values=pad_idx)
        
            batch_R1 = torch.from_numpy(batch_R1)
            batch_R1 = batch_R1.to(device)
            batch_R2 = torch.from_numpy(batch_R2)
            batch_R2 = batch_R2.to(device)
            
            # Negative sampling
            # Take randomly negative samples for every training sample in the current batch, i.e. 
            # Dimension: (batch_size, negative_samples, maxlen).
            # TODO: Allow sampling from different batches.
            
            negative_samples_R1 = torch.stack(tuple(
                [ batch_R1[torch.randperm(jae_abae_conf['batch_size'])[:jae_abae_conf['negative_samples']]] for _ in range(jae_abae_conf['batch_size'])]
                ))
            negative_samples_R2 = torch.stack(tuple(
                [ batch_R2[torch.randperm(jae_abae_conf['batch_size'])[:jae_abae_conf['negative_samples']]] for _ in range(jae_abae_conf['batch_size'])]
                ))
        
            # Reconstruction.
            recon_loss = model(batch_R1, batch_R2, negative_samples_R1, negative_samples_R2)
            
            # Error computation.
            loss = criterion(recon_loss, criterion_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tqdb_batch_iter.set_postfix(Loss=loss.item(),
                                        LR=optimizer.param_groups[0]['lr'])
            
            # Save the model if u wish.
            
        logger.info(f"Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}") #LR = Learning Rate, check optimizer def.
        
    # Print topic-word distribution after each epoch.
    if log_topics:
        for i, topic in enumerate(model.get_topic_words(w2v_model)):
            w = " ".join([t for t in topic])
            logger.info(f'[{i}]\n{w}')
        
        
#%% Save topics

topics_p1 = model.get_topic_words('private_1', w2v_model)
topics_p2 = model.get_topic_words('private_2', w2v_model)
topics_s12 = model.get_topic_words('shared', w2v_model)

save_topics(topics_p1, data_conf['topic_path']+'topics_p1.txt')
save_topics(topics_p2, data_conf['topic_path']+'topics_p2.txt')
save_topics(topics_s12, data_conf['topic_path']+'topics_s12.txt')

        
        
        
        
        
        
        
        