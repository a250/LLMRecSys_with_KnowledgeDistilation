# _*_ coding: utf-8 _*_
# @Time : 2020/10/4
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DCN
################################################
Reference:
    Ruoxi Wang at al. "Deep & Cross Network for Ad Click Predictions." in ADKDD 2017.

Reference code:
    https://github.com/shenweichen/DeepCTR-Torch
"""

import torch
import os
import json
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss

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


class DCNKAR(ContextRecommender):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.

    """

    def __init__(self, config, dataset):
        super(DCNKAR, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.cross_layer_num = config["cross_layer_num"]
        self.reg_weight = config["reg_weight"]
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
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(
                torch.randn((self.num_feature_field + 1)* self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(
                torch.zeros((self.num_feature_field + 1) * self.embedding_size).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [
            self.embedding_size * (self.num_feature_field + 1)
        ] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = (
            self.embedding_size * (self.num_feature_field + 1) + self.mlp_hidden_size[-1]
        )

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.predict_layer = nn.Linear(in_feature_num, 1)
        self.reg_loss = RegLoss()
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

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]

        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def forward(self, interaction):
        dcn_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]

        # KAR Section begin
        user_emb = self.user_profile[interaction[self.user_id]]
        item_emb = self.item_profile[interaction[self.item_id]]
        kar_output = self.kar_layer(user_emb, item_emb)
        kar_output = kar_output.unsqueeze(1)
        dcn_all_embeddings = torch.cat((dcn_all_embeddings, kar_output), dim = 1)        
        # KAR Section end        

        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(batch_size, -1)

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        l2_loss = self.reg_weight * self.reg_loss(self.cross_layer_w)
        return self.loss(output, label) + l2_loss

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
