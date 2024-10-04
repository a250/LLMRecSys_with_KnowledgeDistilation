import os
import importlib
import time
import json
from typing import List

import torch
from logging import getLogger
from copy import deepcopy
from sklearn.decomposition import PCA

from recbole.data import (
    create_dataset, 
    data_preparation, 
    get_dataloader, 
    save_split_dataloaders, 
    load_split_dataloaders,
    create_samplers
)

from recbole.utils import (
    get_model as recbole_get_model,
    ModelType,
    set_color
)

from recbole.model.abstract_recommender import GeneralRecommender


def get_model(model_name):
    try:
        return recbole_get_model(model_name)
    except ValueError:
        # If model not found in RecBole, try to import from models.py
        models_module = importlib.import_module(f'models.{model_name.lower()}')
        if hasattr(models_module, model_name):
            return getattr(models_module, model_name)
        else:
            raise ValueError(f"Model {model_name} not found in RecBole or models.py")


def get_loss(loss_name):
    losses_module = importlib.import_module('losses')
    if hasattr(losses_module, loss_name):
        return getattr(losses_module, loss_name)
    else:
        raise ValueError(f"Loss {loss_name} not found in losses.py")


def process_llm_emb(config, dataset):

    def exp_2024_04_11(from_file, to_file):        
        
        with open(from_file) as f:
            full_descr = json.load(f)
            
        full_dict = {descr['id']:descr['embedding'] for descr in full_descr}
        full_dict = sorted(full_dict.items(), key=lambda x: x[0])
        full_dict = [emb[1] for emb in full_dict]
        user_embs = torch.tensor(full_dict)  
        print(f'getting from: {from_file}')
        print(f'   saving to: {to_file}')

        torch.save(user_embs, to_file)


    def exp_2024_05_18_AVC_users(from_file, to_file):        
        
        with open(from_file) as f:
            full_descr = json.load(f)
            
        user_id = config.USER_ID_FIELD
        # dataset.field2id_token['user_id'][1:]
        full_dict = {descr['id']:descr['embedding'] for descr in full_descr}
        recbole_users_order = dataset.field2id_token[user_id][1:]
        
        # full_dict = sorted(full_dict.items(), key=lambda x: x[0])
        full_dict = [full_dict[k] for k in recbole_users_order]
        # full_dict = [emb[1] for emb in full_dict]
        user_embs = torch.tensor(full_dict)  
        print(f'getting from: {from_file}')
        print(f'   saving to: {to_file}')

        torch.save(user_embs, to_file)        
        
        
    llm_config = config['llm_config']
    llm_preproc_cfg = llm_config['preprocess_from_json']
    llm_random_cfg  = llm_config['preprocess_random']
        
    embeddings_path = llm_config['embeddings_path']

    # path: 
    #     llm_embeddings_path/
    #           dataset_name/
    #.               users_emb_file
    # vars: $embeddings_path/$dataset.dataset_name/$users_emb_file
          
    
    if llm_random_cfg['random']:
        
        random_users_emb_file = llm_random_cfg['random_users_emb_file']
        random_items_emb_file = llm_random_cfg['random_items_emb_file']
        
        random_users_embeddings_path = os.path.join(
            embeddings_path, 
            os.path.join(dataset.dataset_name, random_users_emb_file)
        )        

        random_items_embeddings_path = os.path.join(
            embeddings_path, 
            os.path.join(dataset.dataset_name, random_items_emb_file)
        )          

        if os.path.exists(random_users_embeddings_path) or os.path.exists(random_items_embeddings_path):
            print(f'Nothing done: `{random_users_embeddings_path}` or `{random_items_embeddings_path}` already exists')
            return         
        
        
        n_users = dataset.user_num
        n_items = dataset.item_num
        embedding_size = llm_random_cfg['embedding_size']

        llm_users_embeddings = torch.randn(n_users, embedding_size)
        llm_items_embeddings = torch.randn(n_items, embedding_size)

        torch.save(llm_users_embeddings, random_users_embeddings_path)
        torch.save(llm_items_embeddings, random_items_embeddings_path)

    else:
        
        llm_by   = llm_preproc_cfg['by']
        llm_from = llm_preproc_cfg['from']
        llm_to   = llm_preproc_cfg['to']

        from_file = os.path.join(
            embeddings_path, 
            os.path.join(dataset.dataset_name, llm_from)
        )

        to_file = os.path.join(
            embeddings_path, 
            os.path.join(dataset.dataset_name, llm_to)
        )

#         if os.path.exists(to_file):
#             print(f'Nothing done: `{to_file}` already exists')
#             return            
         
        if llm_preproc_cfg['by'] == 'exp_2024_04_11':
            exp_2024_04_11(from_file, to_file)
        elif llm_preproc_cfg['by'] == 'exp_2024_05_18_AVC_users':
            exp_2024_05_18_AVC_users(from_file, to_file)
        
         
def save_user_item_id(dataset, fname='', version=1):
        json_data = ({
                        'version': f'{version}',
                        'dataset': f'{dataset.dataset_name}',
                        'filters': {
                            'user_inter_num_interval':'[20,500)',
                            'item_inter_num_interval':'[10,inf)'
                        },
                        'item_id': list(dataset.field2id_token['item_id'])[1:],
                        'user_id': list(dataset.field2id_token['user_id'])[1:]
                    })
        
        if fname != '':
            fname = fname + '_'

        with open(f'{dataset.dataset_path}/{dataset.dataset_name}_{fname}{time.strftime("%Y-%m-%d_%H-%M")}.json', 'w') as fp:
            json.dump(json_data, fp)
            
def load_user_item_id(fname):
    with open(f'./{fname}') as f:
        d = json.load(f)
    return d
    
    
def advanced_data_preparation(config, dataset):
    """
    Depends on 
    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.    
    or
        tuple:
            - train_data_prt1 (AbstractDataLoader): The dataloader for training.
            - train_data_prt2 (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.    
    """
    
    if config['add_train_split']:
        
        train_prt1, train_prt2, train_full, valid, test = _train_splitter_data_preparation(config, dataset)
        return [train_prt1, train_prt2, train_full], valid, test
    
    else:
        train, valid, test = data_preparation(config, dataset)
        return [train], valid, test



def _train_splitter_data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuples:
             (
                 ( 
                     train_data_prt1 : AbstractDataLoader,
                     train_data_prt2 : AbstractDataLoader,
                     train_data_full : AbstractDataLoader
                 ),                                    # The dataloaders for training.
                 valid_data : AbstractDataLoader,      # The dataloader for validation.
                 test_data : AbstractDataLoader        # The dataloader for testing.   
             ), 
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset_full, valid_dataset, test_dataset = built_datasets
        train_sampler_full, valid_sampler, test_sampler = create_samplers(
            config, dataset, [train_dataset_full, valid_dataset, test_dataset]
        )

        # in setup of spliting train - return two parts 
        # add_train_shuffle: False
        # add_train_split: [0.5, 0.5]
        train_dataset = deepcopy(train_dataset_full)
        if config['add_train_shuffle']:
            train_dataset.shuffle()
        
        group_by = None
        if config['eval_args']['group_by'] == 'user':
            group_by = dataset.uid_field
        
        train_dataset_prt1, train_dataset_prt2 = train_dataset.split_by_ratio(config['add_train_split'], group_by = group_by)
        # train_dataset_prt1, train_dataset_prt2 = train_dataset, train_dataset

        train_sampler_prt1, _, _ = create_samplers(
            config, dataset, [train_dataset_prt1, valid_dataset, test_dataset]
        )

        train_sampler_prt2, _, _ = create_samplers(
            config, dataset, [train_dataset_prt2, valid_dataset, test_dataset]
        )        
        

        if model_type != ModelType.KNOWLEDGE:
            train_data_full = get_dataloader(config, "train")(
                config, train_dataset_full, train_sampler_full, shuffle=config["shuffle"]
            )          

            train_data_prt1 = get_dataloader(config, "train")(
                config, train_dataset_prt1, train_sampler_prt1, shuffle=config["shuffle"]
            )
            
            train_data_prt2 = get_dataloader(config, "train")(
                config, train_dataset_prt2, train_sampler_prt2, shuffle=config["shuffle"]
            )          
            

            
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )
            print('Alert! Check realization for knowledge models!!!')

        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(
                    train_data_prt1,
                    train_data_prt2,                    
                    valid_data, 
                    test_data
                )
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data_prt1, train_data_prt2, train_data_full, valid_data, test_data


