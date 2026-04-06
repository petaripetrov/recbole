import torch

from recbole.model.abstract_recommender import MACRRecommender
from recbole.model.general_recommender._mf import _MF
from recbole.model.init import xavier_normal_initialization
from recbole.utils.enum_type import InputType

import torch.nn as nn

class MF_MACR(MACRRecommender, _MF):
    r"""
        MF implementation of MACR
    """
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.loss = nn.BCELoss()
        
        self.apply(xavier_normal_initialization)
        
    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        yk = torch.mul(user_e, item_e).sum(dim=1)
        yu = self.sigmoid(self.user_module(user_e)).squeeze(-1)
        yi = self.sigmoid(self.item_module(item_e)).squeeze(-1)
        yui = self.sigmoid(yk * yu * yi)

        return yk, yui, yu, yi
    
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        yk, yui, yu, yi = self.forward(user, item)
        loss_o = self.loss(yui, label)
        loss_i = self.loss(yi, label)
        loss_u = self.loss(yu, label)
        loss = loss_o + self.item_loss_weight * loss_i + self.user_loss_weight * loss_u

        return loss