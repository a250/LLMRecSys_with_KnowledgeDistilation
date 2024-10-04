from .losses import DistilLosses

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class UserEmbeddings(nn.Module):
    """Vector representation of users."""
    
    def __init__(
        self, 
        user_dim: int, 
        embedding_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 16,
    ):
        """Initialize the user embeddings.  
        
        Args:
            user_dim (int): Number of features used to represent the user.
            embedding_dim (int): Dimension of the user embedding.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layer.
        
        """
        super().__init__()
        
#         self.seq = nn.Sequential(
#             nn.Linear(user_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embedding_dim),
#         )
        self.user_embedding = nn.Embedding(user_dim, embedding_dim)
        
    def forward(self, x) -> torch.Tensor:
        """Get the user embeddings.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, user_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        # print(x, x.shape)
#         x = torch.reshape(x.type('torch.FloatTensor'), shape=(-1,1))
#         output = self.seq(x)
        output = self.user_embedding(x)
        return output
    

class ItemEmbeddings(nn.Module):
    """Vector representation of items (films)."""
    
    def __init__(
        self, 
        item_dim: int, 
        embedding_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 16,
    ):
        """Initialize the item embeddings.
        
        Args:
            item_dim (int): Number of features used to represent the item.
            embedding_dim (int): Dimension of the item embedding.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__()
        
#         self.seq = nn.Sequential(
#             nn.Linear(item_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embedding_dim),
#         )            
        self.item_embedding = nn.Embedding(item_dim, embedding_dim)

        
    def forward(self, x) -> torch.Tensor:
        """Get the item embeddings.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        
#         x = torch.reshape(x.type('torch.FloatTensor'), shape=(-1,1))
#         output = self.seq(x)
        output = self.item_embedding(x)
        
        
        return output 
    
    
class Joint(nn.Module):
    """Joint representation of users and items."""
    
    # Default values for the hidden dimension and dropout rate.
    _DEFAULT_HIDDEN_DIM = 16
    _DEFAULT_DROPOUT = 0.2
    
    @classmethod
    def get_default_hidden_dim(cls):
        return cls._DEFAULT_HIDDEN_DIM
    
    @classmethod
    def get_default_dropout(cls):
        return cls._DEFAULT_DROPOUT 
    
    def __init__(
        self, 
        emb_dim, 
        joint_dim,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = 0.2,
    ):
        """Initialize the layer for the joint representation.
        
        Args:
            emb_dim (int): Dimension of the user and item embeddings.
            joint_dim (int): Dimension of the joint representation.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.item = nn.Linear(emb_dim, joint_dim)
        self.user = nn.Linear(emb_dim, joint_dim)
        
        # !!! Сомнительная реализация. Не согласованы размерности
        
        self.ff_head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_dim),
        )
        
        
    def forward(self, user, item) -> torch.Tensor:
        """
            !!!! Крайне сомнительная реализация. 
            Заявлено что происходит конкатенация,  по факту - сложение
        """
        
        """Get the joint representation.
        
        Average the item embeddings and concatenate them with the user embeddings
        by summing them. Then, pass the result through a feed-forward head.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, emb_dim).
            item_embeddings (torch.Tensor): Tensor of shape (batch_size, num_items, emb_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, joint_dim).
        """

        # print(f'{self.user(user).shape =} | {self.item(item).shape =} | {(self.user(user) + self.item(item)).shape =}')
        
        
        return self.ff_head(self.user(user) + self.item(item))
        # return self.ff_head(self.user(user) * self.item(item))        
        # return self.ff_head(torch.mm(self.user(user), torch.transpose(self.item(item),0, 1)))
    

class RecModel(nn.Module):
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embedding_dim: int,
        joint_dim: int,
        dropout: float,
        hidden_dim: int,
    ):
        """Initialize the model.
        
        Args:
            num_user_hidden_layers (int): Number of hidden layers in the user block.
            num_item_hidden_layers (int): Number of hidden layers in the item block.
            user_dim (int): Number of features used to represent the user.
            item_dim (int): Number of features used to represent the item.
            embedding_dim (int): Dimension of the user and item embeddings. 
                Assumes that the user and item embeddings have the same dimension.
            joint_dim (int): Dimension of the joint representation.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super().__init__()
        
        # Multi-layer perceptron for the user embeddings.
        self.user_block = UserEmbeddings(
            user_dim=user_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        # Multi-layer perceptron for the item embeddings.
        self.item_block = ItemEmbeddings(
            item_dim=item_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        # Layer for the joint representation.
        self.joint = Joint(
            emb_dim=embedding_dim,
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
    def forward(self, user, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the joint representation.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, user_dim).
            items (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            tuple: Tuple of tensors of shapes:
                - (batch_size, embedding_dim): User embeddings.
                - (batch_size, embedding_dim): Item embeddings.
                - (batch_size, joint_dim): Joint representation.
        """
        user = self.user_block(user)
        item = self.item_block(item)
        return user, item, self.joint(user, item)
    
    

class RecRegressionHead(nn.Module):
    """Model with classification head. Predicts whether the user likes the item or not."""
    
    def __init__(
        self, 
        joint_dim: int,
    ):
        """Initialize the model.
        
        Args:
            joint_dim (int): Dimension of the joint representation.
        """
        super().__init__()
        
        # Deep neural network for the classification head.
        self.head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(),
            nn.Linear(joint_dim // 2, joint_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(joint_dim // 4),
            nn.Linear(joint_dim // 4, joint_dim // 16),
            nn.ELU(),
            nn.Linear(joint_dim // 16, 1),
        )

    def forward(self, j) -> torch.Tensor:
        """Get the predicted ratings.
        
        Args:
            j (torch.Tensor): Tensor of shape (batch_size, joint_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1).
        """
        return self.head(j)    
    
    
class RecClassificationHead(nn.Module):
    """Model with classification head. Predicts whether the user likes the item or not."""
    
    def __init__(
        self, 
        joint_dim: int,
    ):
        """Initialize the model.
        
        Args:
            joint_dim (int): Dimension of the joint representation.
        """
        super().__init__()
        
        # Deep neural network for the classification head.
        self.head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(),
            nn.Linear(joint_dim // 2, joint_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(joint_dim // 4),
            nn.Linear(joint_dim // 4, joint_dim // 16),
            nn.ELU(),
            nn.Linear(joint_dim // 16, 1),
        )

    def forward(self, j) -> torch.Tensor:
        """Get the predicted ratings.
        
        Args:
            j (torch.Tensor): Tensor of shape (batch_size, joint_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1).
        """
        return self.head(j)

    
# class RecModelLightning(L.LightningModule):
class TwoTowers(GeneralRecommender):
    # input_type = InputType.PAIRWISE
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        super(TwoTowers, self).__init__(config, dataset)

        self.LABEL = config["LABEL_FIELD"]        
        self.RATING = config["RATING_FIELD"]        
        
        
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        
#         user_dim = 1 # dataset.user_num
#         item_dim = 1 # dataset.item_num
        
        embedding_dim = config['embedding_dim']
        joint_dim = config['joint_dim']
        dropout = config['dropout']
        hidden_dim = config['hidden_dim']
            
#     def __init__(
#         self,
#         user_dim: int,
#         item_dim: int,
#         embedding_dim: int,
#         joint_dim: int,
#         dropout: float,
#         hidden_dim: int,
#         lr: float,
#         weight_decay: float,
#         gamma: float = 1E-4,
#     ):
#         """Initialize the model.
        
#         Args:
#             user_dim (int): Number of features used to represent the user.
#             item_dim (int): Number of features used to represent the item.
#             embedding_dim (int): Dimension of the user and item embeddings. 
#                 Assumes that the user and item embeddings have the same dimension.
#             joint_dim (int): Dimension of the joint representation.
#             dropout (float): Dropout rate.
#             hidden_dim (int): Dimension of the hidden layers.
#             lr (float): Learning rate.
#             weight_decay (float): Weight decay.
#             gamma (float): Weight of the vector loss.
#         """
#         super().__init__()
        
        # self.save_hyperparameters()
        
        self.model = RecModel(
            user_dim=self.n_users,
            item_dim=self.n_items,
            embedding_dim=embedding_dim,
            joint_dim=joint_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        self.output = RecRegressionHead(joint_dim)
       
        # self.output = RecClassificationHead(joint_dim)
        
        self.output_loss = nn.MSELoss()
        # self.output_loss = nn.BCEWithLogitsLoss()
        self.vec_cos = nn.CosineSimilarity(dim=1)
        
    def forward(
        self,
        user: torch.Tensor,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the predicted ratings.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, user_dim).
            items (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            tuple: Tuple of tensors of shapes:
                - (batch_size, 1): Predicted ratings.
                - (batch_size, embedding_dim): User embeddings.
                - (batch_size, embedding_dim): Item embeddings.
        """
        
        user, items, j = self.model(user, items)
      
        y = self.output(j)
    
        
        return y # , items, user 
    
    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL] 
        rating = interaction[self.RATING] 
        
        output = self.forward(user, item)        
        #loss = self.output_loss(output, label)
        loss = self.output_loss(output, label)        
        
#         user = interaction[self.USER_ID]
#         pos_item = interaction[self.ITEM_ID]
#         neg_item = interaction[self.NEG_ITEM_ID]
#         user_e = self.user_embedding(user)
#         pos_item_e = self.item_embedding(pos_item)
#         neg_item_e = self.item_embedding(neg_item)

#         pos_interaction = torch.cat((user_e, pos_item_e), -1)
#         neg_interaction = torch.cat((user_e, neg_item_e), -1)

#         pos_item_score = self.MLP_layers(pos_interaction).squeeze()
#         neg_item_score = self.MLP_layers(neg_interaction).squeeze()

#         mf_loss = self.loss(pos_item_score, neg_item_score)

        return loss
        
    
    def predict(self, interaction):
        
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        
        return output
        
class DistilTT_loss(DistilLosses):
    """Toy-example of calculating Cosinus-similarity for ForDistilNCF. Distil on outputs of layers#0 & 1:
        {
            'total_params': 287901,
            'train_params': 287901,
            'layers': [
                (0, Embedding(944, 100)),
                (1, Embedding(1683, 100)),
                (2, BPRLoss()),
                (3, Sequential(
                     (0): Linear(in_features=200, out_features=100, bias=True)
                     (1): ReLU()
                     (2): Linear(in_features=100, out_features=50, bias=True)
                     (3): ReLU()
                     (4): Linear(in_features=50, out_features=1, bias=True)
               ))
            ]
        }
    """ 
    
    def __init__(self, llm_path):
        super(DistilTT_loss, self).__init__(llm_path)
        
        
    def __call__(self, hidden_out, interaction):
        self.hidden_out = hidden_out
        self.interaction = interaction
        
        usr_lr, itm_lr = self.hidden_out.keys()
        user_e = self.hidden_out[usr_lr]['output'][0]
        pos_item_e, neg_item_e = self.hidden_out[itm_lr]['output']
        
        user_id = interaction[self.model.USER_ID]        
        pos_item = interaction[self.model.ITEM_ID]
        neg_item = interaction[self.model.NEG_ITEM_ID]        
       
        
        loss = (
            F.cosine_similarity(user_e, self.llm_user_emb[user_id]).abs().sum()
            + F.cosine_similarity(pos_item_e, self.llm_item_emb[pos_item]).abs().sum()
            + F.cosine_similarity(neg_item_e, self.llm_item_emb[neg_item]).abs().sum()
        )
        return loss