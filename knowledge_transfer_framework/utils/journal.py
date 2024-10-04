import os
from time import strftime
from pathlib import PurePath
import yaml
import pandas as pd

from recbole.config import Config
from .utils import get_model

class ExperimentsJournal:
    def __init__(self, config_file: PurePath, dataset):
        self.config_file  = config_file
        self.parent_dir   = config_file.parent
        self.config_fname = self.config_file.name
        self.exp_subdir   = '/'.join(PurePath(self.parent_dir).parts[1:])
        
        # 'experiments/neumf/exp_0_NeuMF_ml1m.yaml'
    
        self.start_moment = strftime('%Y-%m-%d %H:%M:%S')
        self.end_moment = None

        self.start_date, self.start_time = self.start_moment.split(' ')
        self.end_date, self.end_time = None, None
        
        self.dataset = dataset
        self.config = None
        self._process_config()        
        self.jstruct = self._init_struc()


        self.journal_name = self.config['journal_name']        
        journal_dir = self.parent_dir # os.path.join(self.exp_dir, self.exp_subdir)
        self.journal_csv = os.path.join(journal_dir, self.journal_name)        
        self._init_journal()        
        
        self.train_part = None      

        self.jstruct = self._init_struc()
        
        
    def _init_struc(self):
        struc = {
            'subdir': self.exp_subdir,
            'config': self.config_fname,
            'start date': self.start_date,
            'start time': self.start_time,
            'end date': None,
            'end time': None,         
            'dataset': self.dataset.dataset_name,
            'user_num': self.dataset.user_num,
            'item_num': self.dataset.item_num,
            'inter_num': self.dataset.inter_num,
            'seed': self.config['seed'],            
            'user_filter': self.config['user_inter_num_interval'],
            'item_filter': self.config['item_inter_num_interval'],
            'threshold': self.config['threshold'],
            'model_type': str(self.config['MODEL_INPUT_TYPE']),
            'train_split': self.config['add_train_split'],
            'eval_split': self.config['eval_args']['split'],
            'eval_valid_mode': self.config['eval_args']['mode']['valid'],
            'eval_test_mode': self.config['eval_args']['mode']['test'],
            'eval_valid_metric': self.config['valid_metric'],
            'dist_loss_w': self.config['distil_loss_weight'],
            'train_0:loss_name': None,
            'train_0:users_emb': None,
            'train_0:items_emb': None,
            'train_0:particular': None,
            'train_0:set_outputs': None,
            'train_0:set_trainable': [],
            'train_0:train_ds_ind': None,
            'train_0:config': [],
            'train_0:best_val_score': None,
            'train_0:best_val_res': None,
            'train_0:best_test_res': None,
            'train_0:bestmodel_file': None,            
            'train_1:loss_name': None,
            'train_1:users_emb': None,
            'train_1:items_emb': None,
            'train_1:particular': None,
            'train_1:set_outputs': None,
            'train_1:set_trainable': [],
            'train_1:train_ds_ind': None,
            'train_1:config': [],
            'train_1:best_val_score': None,
            'train_1:best_val_res': None,
            'train_1:best_test_res': None, 
            'train_1:bestmodel_file': None,
        }
        return struc
    
    def catch_instruction(self, cmd, params):
        
        catch = {
#            'train': self._next_train,
            'wrap_model': self._set_distil_loss,
            'set_trainable': self._set_trainable,
            'set_outputs': self._set_outputs,
            'set_train_dataset': self._set_train_dataset,
            'set_config': self._set_config,
            'set': self._set_params,
        }
        
        if cmd in catch.keys():
            catch[cmd](params)
            
    def _set_end_moment(self):
        self.end_moment = strftime('%Y-%m-%d %H:%M:%S')
        self.end_date, self.end_time = self.end_moment.split(' ')
        self.jstruct['end date'] = self.end_date
        self.jstruct['end time'] = self.end_time


    def _set_distil_loss(self, params):
        
        self.jstruct[f'train_{self.train_part}:loss_name'] = params.get('distil_loss', None)
        self.jstruct[f'train_{self.train_part}:users_emb'] = params.get('users_emb_file', None)
        self.jstruct[f'train_{self.train_part}:items_emb'] = params.get('items_emb_file', None)
        self.jstruct[f'train_{self.train_part}:particular'] = params.get('particular', None)
    
    def _set_outputs(self, params):
        self.jstruct[f'train_{self.train_part}:set_outputs'] = params

    def _set_trainable(self, params):
        self.jstruct[f'train_{self.train_part}:set_trainable'] += params
    
    def _set_train_dataset(self, params):
        self.jstruct[f'train_{self.train_part}:train_ds_ind'] = params
    
    def _set_config(self, params):
        self.jstruct[f'train_{self.train_part}:config'] += [params]
        
    def set_val_results(self, best_score, best_result):
        self.jstruct[f'train_{self.train_part}:best_val_score'] = best_score
        self.jstruct[f'train_{self.train_part}:best_val_res'] = best_result

    def set_test_results(self, best_result):
        self.jstruct[f'train_{self.train_part}:best_test_res'] = best_result

    def set_bestmodel_file(self, modelfile):
        self.jstruct[f'train_{self.train_part}:bestmodel_file'] = modelfile

        
        
    def _set_params(self, params):
        train_part = params.get('train_part', None)
        if train_part is not None:
            self.train_part = train_part
        
    def _process_config(self):
        with open(self.config_file) as file:
            config_dict = yaml.safe_load(file)
        model_class = get_model(config_dict["model"])
        self.config = Config(model=model_class, config_dict=config_dict)
        

    def _init_journal(self):        
        if not os.path.exists(self.journal_csv):
            journal = pd.DataFrame({k:[v] for k, v in self.jstruct.items()})
            journal.to_csv(self.journal_csv)
            
        self.journal = pd.read_csv(self.journal_csv, index_col=0)
    
    def save_results(self):
        self._set_end_moment()
        
        row = ({k:[v] for k, v in self.jstruct.items()})
        self._journal = pd.concat([self.journal, pd.DataFrame(row)]).reset_index(drop=True)
        
        self._journal.to_csv(self.journal_csv)
