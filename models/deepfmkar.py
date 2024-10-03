# -*- coding: utf-8 -*-
# @Time   : 2020/7/8
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : deepfm.py

# UPDATE:
# @Time   : 2020/8/14
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
DeepFM
################################################
Reference:
    Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." in IJCAI 2017.
"""

import torch.nn as nn
import torch
import json
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers

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


class DeepFMKAR(ContextRecommender):
    """DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
    Also DeepFM can be seen as a combination of FNN and FM.

    """

    def __init__(self, config, dataset):
        super(DeepFMKAR, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]


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

        self.kar_layer = KARModule(
            user_dim=self.user_kar_emb, 
            item_dim=self.item_kar_emb, 
            output_dim=self.ouput_kar_dim, 
            experts_num=self.item_kar_experts_n
        )
        
        # self.user_profile = torch.load(self.user_profile_file)
        # self.item_profile = torch.load(self.item_profile_file)
        self.user_profile = load_profile_from_file(
            f"{config['data_path']}/{self.user_profile_file}",
            dataset,
            entity_id=self.user_id
        )

        self.item_profile = load_profile_from_file(
            f"{config['data_path']}/{self.item_profile_file}",
            dataset,
            entity_id=self.item_id
        )

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * (self.num_feature_field + 1) # KAR additional feature
        ] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1
        )  # Linear product to the final score
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

    def forward(self, interaction):
        deepfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]

        # KAR Section begin
        user_emb = self.user_profile[interaction[self.user_id]]
        item_emb = self.item_profile[interaction[self.item_id]]
        kar_output = self.kar_layer(user_emb, item_emb)
        kar_output = kar_output.unsqueeze(1)
        deepfm_all_embeddings = torch.cat((deepfm_all_embeddings, kar_output), dim = 1)        
        # KAR Section end    

        batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1))
        )
        y = y_fm + y_deep
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
