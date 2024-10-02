import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilLosses:
    def __init__(
        self, 
        model, 
        llm_users_emb_file: str = None, 
        llm_items_emb_file: str = None
    ):      

        self.model = model
        self.llm_config = model.config['llm_config']
                
        self.llm_users_emb_file = (
            llm_users_emb_file if llm_users_emb_file 
            else self.llm_config['default_users_emb_file']
        )

        self.llm_items_emb_file = (
            llm_items_emb_file if llm_items_emb_file 
            else self.llm_config['default_items_emb_file']
        )              
            
        self.llm_user_emb = None
        self.llm_item_emb = None
        self.particular = False  # if embedding will apply particularly even if original output bigger
        
        # self._set_mapping()
        self._load_embeddings()

        
    def set_particular(self, if_particular):
        self.particular = if_particular
        
        
#     def _set_mapping(self):
#         user = self.model.USER_ID
#         item = self.model.ITEM_ID

#         original_usr_ind = torch.tensor(self.model.dataset.field2id_token[user][1:].astype(int))
#         original_itm_ind = torch.tensor(self.model.dataset.field2id_token[item][1:].astype(int))
#         z = torch.tensor([0])
        
#         self.user_map = torch.hstack((z, original_usr_ind))
#         self.item_map = torch.hstack((z, original_itm_ind))

        
    def _load_embeddings(self):
        
        llm_path = self.llm_config['embeddings_path']        
        dataset_llm_path = os.path.join(llm_path, self.model.dataset.dataset_name)
        
        # Load LLM Embeddings from files
        if self.llm_config['use_user_emb']:            
            self.llm_users_embeddings_path = os.path.join(dataset_llm_path, self.llm_users_emb_file)
            self.llm_user_emb = torch.load(self.llm_users_embeddings_path)
            if self.llm_config['users_start_with_1']:
                n = self.llm_user_emb.shape[1]
                zero_user = torch.zeros((1,n))
                self.llm_user_emb = torch.vstack((zero_user, self.llm_user_emb))


        if self.llm_config['use_item_emb']:            
            self.llm_items_embeddings_path = os.path.join(dataset_llm_path, self.llm_items_emb_file)
            self.llm_item_emb = torch.load(self.llm_items_embeddings_path)   
            if self.llm_config['items_start_with_1']:
                n = self.llm_item_emb.shape[1]
                zero_item = torch.zeros((1,n))
                self.llm_item_emb = torch.vstack((zero_item, self.llm_item_emb))

    def _get_part(self, mtrx1, mtrx2):
        """ get only first [:, :m] elements from mtrx1.
        Args:
            mtrx1 - torch.Tensor, strongly presumed - the output from hidden layer 
            mtrx2 - torch.Tensor - LLM embedings
        
        """
        m = None
        if self.particular:
            m = mtrx2.shape[1]        
        return mtrx1[:,:m]
            
        
    def mse(self, mtrx1, mtrx2):
        mtrx1 = self._get_part(mtrx1, mtrx2)
        m, n = mtrx1.shape
        dlt = mtrx1 - mtrx2
        err = ((dlt * dlt).sum())/(m * n)
        
        return err
    
    def rmse(self, mtrx1, mtrx2):
        err = torch.sqrt(self.mse(mtrx1, mtrx2))
        
        return err
    
    def cossim(self, mtrx1, mtrx2):
        mtrx1 = self._get_part(mtrx1, mtrx2)
        
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        many2many = torch.mm(a_norm, b_norm.transpose(0,1))        
        one2one = torch.diagonal(many2many).sum()
        
        return one2one


    
    def __call__(self, hidden_out, interaction):
        """Calculate loss between given tensors. 
        Must be customized in code depending on shape of inputs
        """
        pass

    


class NeumfUserRMSE(DistilLosses):    
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(NeumfUserRMSE, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        item = interaction[self.model.ITEM_ID]
        label = interaction[self.model.LABEL]        
                
        mlp_hidden = list(self.hidden_out.keys())[0]
        output = self.hidden_out[mlp_hidden]['output'][0]
        
        # gmf = torch.mul(user_e, item_e)        
        # gmf_llm = torch.mul(self.llm_user_emb[user], self.llm_item_emb[item])
        
        # llm_users =  self.user_map[user].long()
        
        # loss = self.rmse(output, self.llm_user_emb[llm_users])
        loss = self.rmse(output, self.llm_user_emb[user])
        
        
        return loss
    
class AutoIntUserRMSE(DistilLosses): 
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(AutoIntUserCosSim, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        item = interaction[self.model.ITEM_ID]
        label = interaction[self.model.LABEL]        

        mlp_hidden = list(self.hidden_out.keys())[0]
        hidden = self.hidden_out[mlp_hidden]['input'][0][0]
        
        # llm_users =  self.user_map[user].long()

        #loss = self.rmse(hidden, self.llm_user_emb[llm_users])
        loss = self.rmse(hidden, self.llm_user_emb[user])


        return loss


class MultiVAEUserRMSE(DistilLosses):    
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(MultiVAEUserRMSE, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        #item = interaction[self.model.ITEM_ID]
        #label = interaction[self.model.LABEL]        
                
        mlp_hidden = list(self.hidden_out.keys())[0]
        output = self.hidden_out[mlp_hidden]['input'][0][0]
        
        # gmf = torch.mul(user_e, item_e)        
        # gmf_llm = torch.mul(self.llm_user_emb[user], self.llm_item_emb[item])
        
        # llm_users =  self.user_map[user].long()
        # loss = self.rmse(output, self.llm_user_emb[llm_users])
        loss = self.rmse(output, self.llm_user_emb[user])        
        
        
        return loss
    
class LightGCNUserCosSim(DistilLosses):    
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(LightGCNUserCosSim, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        item = interaction[self.model.ITEM_ID]
#        label = interaction[self.model.LABEL]        

#         mlp_hidden = list(self.hidden_out.keys())[0]
#         hidden = self.hidden_out[mlp_hidden]['input'][0][0]
        
#         llm_users =  self.user_map[user].long()

#         loss = self.rmse(hidden, self.llm_user_emb[llm_users])


        return 0.1 # loss


class SimpleXUserRMSE(DistilLosses):    
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(SimpleXUserRMSE, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        #item = interaction[self.model.ITEM_ID]
        #label = interaction[self.model.LABEL]        
                
        mlp_hidden = list(self.hidden_out.keys())[0]
        output = self.hidden_out[mlp_hidden]['input'][0][0]
        
        # gmf = torch.mul(user_e, item_e)        
        # gmf_llm = torch.mul(self.llm_user_emb[user], self.llm_item_emb[item])

        #llm_users =  self.user_map[user].long()        
        #loss = self.rmse(output, self.llm_user_emb[llm_users])
        loss = self.rmse(output, self.llm_user_emb[user])
        
        return loss


class NGCFUserRMSE(DistilLosses):    
    def __init__(
        self, 
        model, 
        llm_users_emf_file: str = None, 
        llm_items_emf_file: str = None
    ):
        super(NGCFUserRMSE, self).__init__(model, llm_users_emf_file, llm_items_emf_file)
        
        self.n_users = model.dataset.user_num 
        self.n_items = model.dataset.item_num
        
        
    def __call__(self, hiddens_out, interaction):
        self.hiddens_out = hiddens_out
        self.interaction = interaction
        
        user = interaction[self.model.USER_ID]
        #item = interaction[self.model.ITEM_ID]
        #label = interaction[self.model.LABEL]        
                
        layer_name = list(self.hiddens_out.keys())[0]
        hidden_out = self.hiddens_out[layer_name]['output'][0] # [0]

        user_all_embeddings, item_all_embeddings = torch.split(
            hidden_out, [self.n_users, self.n_items]
        )            
        u_embeddings = user_all_embeddings[user]
        
#         llm_users =  self.user_map[user].long()
#         loss = self.rmse(u_embeddings, self.llm_user_emb[llm_users])
        loss = self.rmse(u_embeddings, self.llm_user_emb[user])
        
        return loss