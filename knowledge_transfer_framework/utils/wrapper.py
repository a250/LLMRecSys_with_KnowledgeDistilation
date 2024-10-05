import os
import importlib
import time
import json
from typing import List

import torch
from logging import getLogger

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

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.trainer import Trainer

class ModelWrapper(AutoEncoderMixin, GeneralRecommender):
    """Provide additional method for model preparation:
        - managing trainable parameters on each layers
        - maneging additional outputs from hidden layers
    
    Args:
        model: instance GeneralRecommender of model 
        distil_loss: instance of DistilLosses customized for calculate losses between 
            model's hidden layers output and given llm embeddings    
    """
    
    def __init__(
        self, 
        model: GeneralRecommender, 
        distil_loss_name: str, 
        llm_users_emb_file: str = None, 
        llm_items_emb_file: str = None
    ):
        super(ModelWrapper, self).__init__(model.config, model.dataset)

        self.model = model        
        self.model_struc = dict()
        

        # load dataset info
        self.model.USER_ID = model.config["USER_ID_FIELD"]
        self.model.ITEM_ID = model.config["ITEM_ID_FIELD"]
        self.model.NEG_ITEM_ID = model.config["NEG_PREFIX"] + self.model.ITEM_ID
        
        
        self.distil_loss_name = distil_loss_name 
        self.llm_users_emb_file = llm_users_emb_file
        self.llm_items_emb_file = llm_items_emb_file        
        self.distil_loss = self._init_distil_loss()
        
        
        self.is_hidden_outputs = False   # True, if model collect additional input/output from hidden layer
        self.hidden_layer_outs = None   # Dict, for store  input and output data on every `calculate_loss` call:
                                        # {
                                        # 'layer_name': 
                                        #     {
                                        #       'input': [input_1, ... ,input_n],
                                        #       'output': [output_1, ... ,output_n] 
                                        #     }
                                        # }
        self.layers_hook = set() # set of hooks functions

        self.distil_loss_weight = self.model.config["distil_loss_weight"]
 
    def _init_distil_loss(self):
        

        distil_loss_class = self._get_loss(self.distil_loss_name)
        
        distil_loss = distil_loss_class(
            self.model, 
            self.llm_users_emb_file, 
            self.llm_items_emb_file
        ) 
        return distil_loss


        
    def _get_loss(self, loss_name):
        
        model_name = self.model.config['model'].lower()        
        losses_module = importlib.import_module('models.losses')
        losses_class = None
        
        if hasattr(losses_module, loss_name):
            print('..getting loss from models.losses')
            losses_class = getattr(losses_module, loss_name)  

        else:
            try:
                model_module = importlib.import_module(f'models.{model_name}')
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"Module models.{model_name} not found")
                
            if hasattr(model_module, loss_name):
                losses_class = getattr(model_module, loss_name)  
            
        if not losses_class:        
            raise ValueError(f'Loss {loss_name} not found neither in module.losses nor models.{model_name}')
        
        return losses_class



    def calculate_loss(self, interaction):
        """Calculate complex loss, combines model loss and distilation loss
        
        """
        
        if self.is_hidden_outputs:        
            self.hidden_layer_outs = dict()            
            model_loss = self.model.calculate_loss(interaction)
            distil_loss = self.distil_loss(self.hidden_layer_outs, interaction)
            alfa = self.distil_loss_weight
            total_loss = (1 - alfa) * model_loss + alfa * distil_loss    
            return total_loss 
        else:
            return self.model.calculate_loss(interaction) 

        
    def predict(self, interaction):
        return self.model.predict(interaction)

    def full_sort_predict(self, interaction):
        return self.model.full_sort_predict(interaction)
            
    def set_trainable(self, layer_uid: List[int], trainable: bool = True) -> None:    
        """Manage which parameters should be trainable or freezen
        Args:
            layers: list of layers for processing, e.g. `[0, 2, 3]`
            trainable: which value shoud be set as True or False for given layers
        """
        self._travers_model_struc()
        if layer_uid == ['*']:
            layer_uid = list(self.model_struc.keys())
           
        for l_uid in layer_uid:
            print(f'Set -> {trainable} | {l_uid} : {self.model_struc[l_uid]}')
            for param in self.model_struc[l_uid].parameters():
                param.requires_grad = trainable

            


    def set_hidden_output(self, layer_uid: List[int]):
        """Set hook function for grab input and output for given layers
        Args:
            layers: list of layers for processing, e.g. `[0, 2, 3]`        
        """
        self._travers_model_struc()
        
        for l_uid in layer_uid:
            self.is_hidden_outputs = True        
            print(f'Hook -> | {l_uid} : {self.model_struc[l_uid]}')
            layer = self.model_struc[l_uid]
            layer_hook = layer.register_forward_hook(self.__hidden_layer_hook_fn)
            self.layers_hook.add(layer_hook)

            
    def remove_hidden_output(self):
        """Remove all hooks from all model, which was previousely given
        """
        for hook in self.layers_hook:
            hook.remove()
        self.layers_hook = set()
        self.is_hidden_outputs = False        
                    
    def __hidden_layer_hook_fn(self, module, input_, output):
        """Store input and output data of given layer on every `calculate_loss` call in form of dict:
        Args:
            module: name of layer where hook was triggered
            input_: tensors for input and output
            output: tensors for input and output
        {
            'layer_name': 
                 {
                   'input': [input_1, ... ,input_n],
                   'output': [output_1, ... ,output_n] 
                 }
        }        
        """
        if module in self.hidden_layer_outs:
            self.hidden_layer_outs[module]['input'].append(input_)
            self.hidden_layer_outs[module]['output'].append(output)
        else:
            self.hidden_layer_outs[module] = {'input' : [input_], 'output': [output]}

#
#   methods for getting additional info about models
#                
    
    def info(self, verbose=0):
        """Providing details about model
        Args:
            verbose: 
                0: give short information about number of trainable and total parameters
                1: adding information about list of model layers
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        traibanle_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        out = {'total_params' : total_params, 'train_params' : traibanle_params}        
        
        if verbose == 1:
            self._travers_model_struc(verbose=verbose)            
            out['layers'] = self.model_struc
#             out['layers'] = []
#             for i, layer in enumerate(self.model.children()):
#                 out['layers'].append((i, layer) )
        
        return out

    def debug_out_hiddens(self, interaction):
        """To check what outputs will give model on the setted hidden layer

        Args:
            interaction: batch for forward prop
            
        Output:
            (model_loss: real, hidden_layers_outs: dict) or 
            (model_loss: real, None)
            
        """
        if self.is_hidden_outputs:        
            self.hidden_layer_outs = dict()            
            model_loss = self.model.calculate_loss(interaction)
            return (model_loss, self.hidden_layer_outs)
        else:
            return (self.model.calculate_loss(interaction), None)

        
    def _travers_model_struc(self, verbose=0):
        
        def _travers(node_name, node, level = 0, spaces=4):
            margin = "".join(["-"]*level*spaces)
            if len(list(node.named_children()))>0:
                self.l_uid +=1
                self.model_struc[self.l_uid] = node
                if verbose:
                    print(f'{self.l_uid:>5} | {margin} | {node_name} | {node.__class__.__name__} ({node.extra_repr()})')        
                
                for n in node.named_children():
                    _travers(n[0], n[1], level+1)
            else:
                self.l_uid +=1
                self.model_struc[self.l_uid] = node                
                if verbose:
                    print(f'{self.l_uid:>5} | {margin} | {node_name} | {node.__class__.__name__} ({node.extra_repr()})')    

        self.l_uid = 0
        self.model_struc = dict()
        _travers('root', self.model)
              
        
    def debug_out_model_struct(self, verbose=1):
        
        self._travers_model_struc(verbose=verbose)
        return self.model_struc
