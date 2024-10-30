import os
from typing import List

import torch
from logging import getLogger
from tqdm import tqdm
import pandas as pd
import umap
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
from recbole.trainer import Trainer

from .gen_wrapper import GeneralModelWrapper
from .con_wrapper import ContextModelWrapper
from .seq_wrapper import SequentialModelWrapper

from .journal import ExperimentsJournal

class TrainManager:
    def __init__(self, config, model_class, datasets):
        """
        Args:
            config: config 
            module: model
            datasets: dict, in form of:
                {
                    'train': [dataset_1, ..., dataset_n],
                    'valid': valid_dataset,
                    'test': test_dataset
                }                
        """
        self.config = config
        self.model_class = model_class
        self.datasets = datasets

        self.journal = None
        
        self.train_data = datasets[0][0]
        self.valid_data = datasets[1]
        self.test_data = datasets[2]
        
        self.instructions = {
            'init_model': self.__init_model,
            'wrap_model': self.__wrap_model,
            'reduce_dim': self.__reduce_dim,            
            'init_trainer': self.__init_trainer,            
            'set_trainable': self.__set_trainable,
            'set_outputs': self.__set_outputs,
            'remove_outputs': self.__remove_outputs,
            'set_config': self.__set_config,
            'set_train_dataset': self.__set_train_dataset,
            'test_eval': self.__test_eval,
            'train': self.__train,
            'info': self.__info,
            'print': self.__print,
            'debug_model': self.__debug_model,
            'load_from_file': self.__load_model,
            #'set_nothing': self.__set_nothing, 
            'save_datasets': self.__save_datasets,
        }
    
    def __details(self, message):
        # print(f'\t\t\t> {message}')
        print(f'   |                      |> {message}')        
        
    def __init_model(self, params):
        self.model = self.model_class(self.config, self.datasets[0][0].dataset).to(self.config["device"])
        
        self.model.config = self.config
        self.model.dataset = self.datasets[0][0].dataset

    def set_journal(self, journal: ExperimentsJournal):
        self.journal = journal

    def __wrap_model(self, params):
        self.intern_model = self.model
        # model_class(config, train_data_prt1.dataset).to(config["device"])
        distil_loss_name = params['distil_loss']
        particular = params['particular']
        users_embeddings_file = params.get('users_emb_file', None)
        items_embeddings_file = params.get('items_emb_file', None)        
        wrapper_type = self.intern_model.type.name
        
        self.__details(f'using `{wrapper_type}` wrapper')
        
        if wrapper_type == 'GENERAL':
            ModelWrapper = GeneralModelWrapper
        elif wrapper_type == 'CONTEXT':
            ModelWrapper = ContextModelWrapper
        elif wrapper_type == 'SEQUENTIAL':
            ModelWrapper = SequentialModelWrapper
        else:
            raise ValueError(f'Unknown type of model: {wrapper_type}')

        self.model = ModelWrapper(
            self.intern_model, 
            distil_loss_name,
            users_embeddings_file,
            items_embeddings_file
        )
        
        self.model.distil_loss.set_particular(particular)

    def __reduce_dim(self, params):
        
        def pca_reducer(mtrx, n_components):
            pca = PCA(n_components=n_components)
            pca.fit(mtrx)
            reduced_emd = pca.transform(mtrx)            
            return torch.tensor(reduced_emd)

        def umap_reducer(mtrx, n_components):
            umap_reducer = umap.UMAP(n_components)
            reduced_emb = umap_reducer.fit_transform(mtrx)
            return torch.tensor(reduced_emb)
        
        file_in = params['file_in']
        file_out = params['file_out']
        n_components = params['dimension']
        overwrite = params['overwrite']
        method_name = params.get('method', 'umap')
        if method_name == 'umap':
            reducer = umap_reducer
        elif method_name == 'pca':
            reducer = pca_reducer

        self.__details(f'apply `{method_name}` to {file_in} -> [:{n_components}] -> {file_out}')
        
        llm_config = self.config['llm_config']        
        embeddings_path = llm_config['embeddings_path']
        llm_path = os.path.join(embeddings_path, self.model.dataset.dataset_name)              
        emb_from_file = os.path.join(llm_path, file_in)
        emb_to_file = os.path.join(llm_path, file_out)        
        
        
        if os.path.exists(emb_to_file):
            if not overwrite:
                self.__details(f'Nothing done: `{emb_to_file}` already exists and {overwrite=}')
                return 

        original_emb = torch.load(emb_from_file)
        
        reduced_emb = reducer(original_emb, n_components)
       
        torch.save(reduced_emb, emb_to_file)

         
    def __init_trainer(self, params):
        self.trainer = Trainer(self.config, self.model)

    def __debug_model(self, params):
        model_struc = self.model.debug_out_model_struct(verbose=0)
        layer = model_struc[params]
        self.__details(f'parameters output for {layer}')
        self.__details(list(layer.named_parameters()))
        
    def __load_model(self, params):
        self.trainer.resume_checkpoint(params)
        
    def __set_trainable(self, params):
        self.__details(f'{params[0]} -> {params[1]}')
        self.model.set_trainable(params[0], params[1])        

    
    def __set_outputs(self, params):
        self.model.set_hidden_output(params)

    def __remove_outputs(self, params):
        self.model.remove_hidden_output()
    
    def __set_config(self, params):
        for k, v in params.items():
            _prev = self.config[k]
            self.config[k] = v
            self.__details(f'config[{k}]: {_prev} -> {v}')         
    
    def __set_train_dataset(self, params):
        self.__details(f'use train_dataset part_{params}')
        self.train_data = self.datasets[0][params]
    
    def __train(self, params):    
        best_valid_score, best_valid_result = self.trainer.fit(
            self.train_data, 
            self.valid_data, 
            saved=self.config['save'], 
            show_progress=self.config['show_progress']
        )
#        print(f'{best_valid_score = }, {best_valid_result = }')
        if self.journal:
            self.journal.set_val_results(best_valid_score, best_valid_result)
            self.journal.set_bestmodel_file(self.trainer.saved_model_file)
            
        self.__details(f'Best Validation Score : {best_valid_score}')
        self.__details(f'Best Validation Result: {best_valid_result}')
        self.__details(f'Best model saved in: {self.trainer.saved_model_file}')        
    
    def __info(self, params):
        # self.__details(self.model.info(params))
        self.model.info(params)
    
    def __print(self, params):
        self.__details(params)
        
    def __test_eval(self, params):
        test_result = self.trainer.evaluate(self.test_data)
        if self.journal:
            self.journal.set_test_results(test_result)

        self.__details(f'{test_result = }')

    def __save_datasets(self, params=None):
        def _travers_dataset(dataset, part_name):
            res = {
                'batch_id': [],        
                'user_id': [],
                'item_id': [],
                'timestamp': [],
                'label': []
            }

            dataloader_class = dataset.__class__.__name__ # 'TrainDataLoader' for train ds,  'NegSampleEvalDataLoader' for val and test
            self.__details(f'Processing {part_name} from {dataloader_class}')
            
            for n, data in enumerate(tqdm(dataset)):
                # print(n)
                # print(batch)
                if dataloader_class == 'NegSampleEvalDataLoader':
                    data = data[0]
                    
                user_ids = list(data['user_id'].numpy())
                item_ids = list(data['item_id'].numpy())
                time_sts = list(data['timestamp'].numpy())
                labels   = list(data['label'].numpy())

                res['batch_id']  += [n] * data['item_id'].shape[0]        
                res['user_id']   += user_ids
                res['item_id']   += item_ids
                res['timestamp'] += time_sts
                res['label']     += labels
                
            df = pd.DataFrame(res).set_index('batch_id')
            df['timestamp'] = df['timestamp'].astype(int)
            df['label'] = df['label'].astype(int)  
            return df
        
        if self.journal is None:
            raise ValueError('For sake of saving dataset you have to set `journal`')

        _datasets = {
            'train': self.datasets[0][0],
            'val': self.datasets[1], 
            'test': self.datasets[2]
        }
        _dataset_name = self.config.dataset
        _moment = self.journal.start_moment.replace(' ','_').replace(':','-') 
        path_to_config = self.journal.parent_dir
        path_to_datasets = os.path.join(path_to_config, 'datasets')
        if not os.path.isdir(path_to_datasets):
            os.makedirs(path_to_datasets)

        for part_name, v in _datasets.items():
            fname = f'{_dataset_name}_{part_name}_{_moment}.tsv'
            path_to_dataset = os.path.join(path_to_datasets, fname)

            df = _travers_dataset(v, part_name)
            df.to_csv(path_to_dataset, sep='\t')
            self.__details(f'saved: {path_to_dataset}')        

    def catch_instruction(self, cmd, prm):
        if cmd in self.instructions.keys():
            self.instructions[cmd](prm)

    def run(self):
        if not self.config['scenario']:
#             self.train_data = datasets['train'][0]
#             self.valid_data = datasets['valid']
#             self.test_data = datasets['test']
            self.__train()
            test_result = self.trainer.evaluate(self.test_data)
            res = {
                'test_result': test_result,
                'best_valid_score': best_valid_score,
                'best_valid_result': best_valid_result
            }
            return res
        
        print('\nScenario execution:')
        print(f'\nstp|       command        |     output')
        print(f'------------------------------------------------------------------------------------')

        
        for i, step in enumerate(self.config['scenario']):
            cmds = step.keys()
            if 'command' in cmds:
                cmd = step['command']
                prm = step.get('params', None)
            else:
                cmd, prm = list(step.items())[0]
            cmd_show = f'`{cmd.upper()}`'
            print(f'{i: >3}|{cmd_show: >22}|')
            if cmd == 'break':
                break            
                
            self.catch_instruction(cmd, prm)

            if self.journal:
                self.journal.catch_instruction(cmd, prm)
            print(f'------------------------------------------------------------------------------------')
