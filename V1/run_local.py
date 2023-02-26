# Use pytorch_py3.8.8

import sys
sys.path.insert(1, '/home/tuomas/Python/Gradu/Puhti/ABAE_model_basic/Model')

from JointAutoEncoder_AttentionBasedAspectEncoder import JAE_ABAE
from reader import TrainingDataset, get_w2v, get_centroids, save_topics, init_topic_matrices

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import os
import json
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import torch.distributed.run
import torch.distributed.launch

def print_timestamp(msg):
    dt = datetime.now().strftime("%H:%M:%S")
    print(f'{dt} - {msg}')
    
def get_optimizer(opt, model):
    optimizer = None
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters())
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters())
    else:
        raise Exception('Optimizer {} is not supported.'.format(opt))
        
    return optimizer

def cleanup():
    torch.distributed.destroy_process_group()
    
def train(data_conf, model_conf, training_round, local_rank):
    
    # Prepare the dataloader.
    print_timestamp('Constructing the dataset.')
    w2v_model = get_w2v(data_conf['wv_file'])
    dataset = TrainingDataset(data_conf, w2v_model.wv.key_to_index, w2v_model.wv.vectors.shape[0])
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=model_conf['batch_size'], pin_memory=True, num_workers=10, 
                            drop_last=True, shuffle=False, sampler=sampler)
    
    # Set up the model.
    print_timestamp('Constructing the model.')
    T1_p_init, T2_p_init, T12_s_init = init_topic_matrices(data_conf['wv_path'],
                                                           data_conf['R1']['wv_path'],
                                                           data_conf['R2']['wv_path'],
                                                           model_conf)
    model = JAE_ABAE(model_conf,
                 torch.from_numpy(np.vstack([w2v_model.wv.vectors, np.zeros(model_conf['emb_dim'], dtype='float32')])),     # Word embedding vectors + zero vector for padding index.
                 init_topic_matrix_p1=T1_p_init,
                 init_topic_matrix_p2=T2_p_init,
                 init_topic_matrix_s12=T12_s_init
                 )
    
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = get_optimizer(model_conf['optimizer'], model)
    
    # Train the model
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    criterion_target = torch.zeros((model_conf['batch_size'], 1)).cuda()
    
    print_timestamp('Training start')
    for e in range(model_conf['epochs']):
        # If we are using DistributedSampler, we have to tell it which epoch this is.
        dataloader.sampler.set_epoch(e)
        epoch_loss = 0
        for step, batch_R1, batch_R2 in enumerate(dataloader):
            
            # Negative sampling
            # Take randomly negative samples for every training sample in the current batch, i.e. 
            # Dimension: (batch_size, negative_samples, maxlen)
            negative_samples_R1 = torch.stack(tuple(
                [ batch_R1[torch.randperm(model_conf['batch_size'])[:model_conf['negative_samples']]] for _ in range(model_conf['batch_size'])]
                ))
            negative_samples_R2 = torch.stack(tuple(
                [ batch_R2[torch.randperm(model_conf['batch_size'])[:model_conf['negative_samples']]] for _ in range(model_conf['batch_size'])]
                ))
    
            # Prediction.
            recon_loss = model(batch_R1, batch_R2, negative_samples_R1, negative_samples_R2)
            
            # Error computation.
            loss = criterion(recon_loss, criterion_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print_timestamp(f"Epoch {e+1}/{model_conf['epochs']}: Loss = {epoch_loss/len(dataloader)}")
        
    # Save topics. Use only rank 0 model since all the models are the same.
    if local_rank==0:
        topics_p1 = model.get_topic_words('private_1', w2v_model, topn=40)
        topics_p2 = model.get_topic_words('private_2', w2v_model, topn=40)
        topics_s12 = model.get_topic_words('shared', w2v_model, topn=40)
        
        save_topics(topics_p1, model_conf['topic_path']+f'topics_p1_{training_round}.txt', readable=False)
        save_topics(topics_p2, model_conf['topic_path']+f'topics_p2_{training_round}.txt', readable=False)
        save_topics(topics_s12, model_conf['topic_path']+f'topics_s12_{training_round}.txt', readable=False)
        
        save_topics(topics_p1, model_conf['topic_path']+f'topics_pretty_p1_{training_round}.txt', readable=False)
        save_topics(topics_p2, model_conf['topic_path']+f'topics_pretty_p2_{training_round}.txt', readable=False)
        save_topics(topics_s12, model_conf['topic_path']+f'topics_pretty_s12_{training_round}.txt', readable=False)


def log_info():
    print_timestamp(f'Using PyTorch version {torch.__version__}')
    if torch.cuda.is_available():
        print_timestamp(f"Avaiable GPU's: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
    else:
        print_timestamp("No CUDA avaiable.") 
    
        
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--training_rounds', type=int, required=True)
    args = parser.parse_args()
    
    # Load dataset configurations and model configuration.
    data_dir = f'/home/tuomas/Python/Gradu/data_processing/datasets/Amazon review data/Training_data/JAE_ABAE/{args.data_name}/'
    model_dir = '/home/tuomas/Python/Gradu/Puhti/JAE-ABAE_model_basic/Model'
    with open(data_dir + 'data_conf.json') as f:
        data_conf = json.load(f)
    with open(model_dir + 'model_conf_local.json') as f:
        model_conf = json.load(f)
    log_info()
    print_timestamp(f'Data directory: {data_dir}')
    
    # Distributed setup.
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl') # Use the NCCL backend for distributed GPU training
    world_size = dist.get_world_size()
    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)
    
    for tr in range(args.training_rounds):
        print_timestamp(f'Training round {tr+1}/{args.training_rounds}.')
        train(data_conf, model_conf, tr, local_rank)
    
    