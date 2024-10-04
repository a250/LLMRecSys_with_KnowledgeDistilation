# -*- coding: utf-8 -*-
# @Time   : 2020/09/01
# @Author : Shuqing Bian
# @Email  : shuqingbian@gmail.com
# @File   : autoint.py

r"""
AutoInt
################################################
Reference:
    Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
    in CIKM 2018.
"""

import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers

class KARModule(nn.Module):
    def __init__(self,
                 user_dim: int,
                 item_dim: int,
                 output_dim: int,
                 experts_num: int,
                 experts_hidden_dim: list[int] | None = None):
        super(KARModule, self).__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.experts_num = experts_num
        self.output_dim = output_dim
        self.common_experts = nn.ModuleList([Expert(input_dim=user_dim + item_dim,
                                                    hidden_dim=experts_hidden_dim,
                                                    output_dim=output_dim) for _ in range(experts_num)])
        self.gate_layer = nn.Linear(user_dim + item_dim, self.experts_num)

    def forward(self, user_emb, item_emb):
        pair = torch.cat([user_emb, item_emb], dim=-1)
        common_expert_out = [expert(pair) for expert in self.common_experts]
        experts = torch.stack(common_expert_out, dim=-1)
        gate = self.gate_layer(pair).unsqueeze(dim=-1)
        return torch.matmul(experts, gate).squeeze(dim=-1)


class Expert(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: list[int] | None = None
                 ):
        super(Expert, self).__init__()
        architecture = [input_dim]
        if hidden_dim:
            architecture.extend(hidden_dim)
        architecture.append(output_dim)

        layers = []
        for layer in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[layer], architecture[layer + 1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_profile_from_file(from_file, dataset, entity_id=None):
    # by entity presumed `user` or `item`, so
    # entity is a config.USER_ID_FIELD or config.ITEM_ID_FIELD

    if not entity_id:
        # Should implement raise error here
        return None
    
    with open(from_file) as f:
        full_descr = json.load(f)
        
    # dataset.field2id_token['user_id'][1:]
    full_dict = {descr['id']:descr['embedding'] for descr in full_descr}
    recbole_entity_order = dataset.field2id_token[entity_id][1:]
    
    # full_dict = sorted(full_dict.items(), key=lambda x: x[0])
    shifted_emb = [full_dict[k] for k in recbole_entity_order]
    # full_dict = [emb[1] for emb in full_dict]
    shifted_emb = torch.tensor(shifted_emb)
    n = shifted_emb.shape[1]
    zero_row = torch.zeros((1,n))
    entity_embs = torch.vstack((zero_row, shifted_emb))        
    return entity_embs

class AutoIntKAR(ContextRecommender):
    """AutoInt is a novel CTR prediction model based on self-attention mechanism,
    which can automatically learn high-order feature interactions in an explicit fashion.

    """

    def __init__(self, config, dataset):
        super(AutoIntKAR, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config["attention_size"]
        self.dropout_probs = config["dropout_probs"]
        self.n_layers = config["n_layers"]
        self.num_heads = config["num_heads"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.has_residual = config["has_residual"]

        # KAR Section
        # KARModule(user_dim=8, item_dim=8, output_dim=4, experts_num=3)
        self.user_id = config.USER_ID_FIELD
        self.item_id = config.ITEM_ID_FIELD

        self.user_kar_emb = config["user_kar_emb"]
        self.item_kar_emb = config["item_kar_emb"]
        self.item_kar_experts_n = config["item_kar_experts_n"]
        self.ouput_kar_dim = config["output_kar_dim"]
        self.user_profile_file = config["user_profile_json"]
        self.item_profile_file = config["item_profile_json"]

        self.llm_config = config['llm_config']
        llm_path = self.llm_config['embeddings_path']        
        dataset_llm_path = os.path.join(llm_path, dataset.dataset_name)
        user_profile_file = os.path.join(dataset_llm_path, self.user_profile_file)
        item_profile_file = os.path.join(dataset_llm_path, self.item_profile_file)        

        self.kar_layer = KARModule(
            user_dim=self.user_kar_emb, 
            item_dim=self.item_kar_emb, 
            output_dim=self.ouput_kar_dim, 
            experts_num=self.item_kar_experts_n
        )

        
        # self.user_profile = torch.load(self.user_profile_file)
        # self.item_profile = torch.load(self.item_profile_file)
        self.user_profile = load_profile_from_file(
            # f"{config['data_path']}/{self.user_profile_file}",
            user_profile_file,
            dataset,
            entity_id=self.user_id
        )

        self.item_profile = load_profile_from_file(
            # f"{config['data_path']}/{self.item_profile_file}",
            item_profile_file,
            dataset,
            entity_id=self.item_id
        )
        
        # define layers and loss
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        self.embed_output_dim = (self.num_feature_field + 1) * self.embedding_size # Add kar here as an feature
        self.atten_output_dim = (self.num_feature_field + 1) * self.attention_size # Add kar here as an feature
        size_list = [self.embed_output_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_probs[1])
        # multi-head self-attention network
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.attention_size, self.num_heads, dropout=self.dropout_probs[0]
                )
                for _ in range(self.n_layers)
            ]
        )
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        if self.has_residual:
            self.v_res_embedding = torch.nn.Linear(
                self.embedding_size, self.attention_size
            )

        self.dropout_layer = nn.Dropout(p=self.dropout_probs[2])
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def autoint_layer(self, infeature):
        """Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of[batch_size,field_size,embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size,1] .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Residual connection
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        # Interacting layer
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        att_output = self.attn_fc(cross_term) + self.deep_predict_layer(
            self.mlp_layers(infeature.view(batch_size, -1))
        )
        return att_output

    def forward(self, interaction):
        autoint_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        
        # KAR Section begin
        user_emb = self.user_profile[interaction[self.user_id]]
        item_emb = self.item_profile[interaction[self.item_id]]
        kar_output = self.kar_layer(user_emb, item_emb)
        kar_output = kar_output.unsqueeze(1)
        autoint_all_embeddings = torch.cat((autoint_all_embeddings, kar_output), dim = 1)        
        # KAR Section end        

        output = self.first_order_linear(interaction) + self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
