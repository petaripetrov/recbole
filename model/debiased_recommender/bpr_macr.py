import torch

from recbole.model.abstract_recommender import MACRRecommender
from recbole.model.general_recommender._bpr import _BPR
from recbole.model.init import xavier_normal_initialization
from recbole.utils.enum_type import InputType


class BPR_MACR(MACRRecommender, _BPR):
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_e = self.get_user_embedding(user)
        pos_e = self.get_item_embedding(item)
        neg_e = self.get_item_embedding(neg_item)
        
        yk_pos = torch.mul(user_e, pos_e).sum(dim=1)
        yk_neg = torch.mul(user_e, neg_e).sum(dim=1)
        
        yu_logit = self.user_module(user_e).squeeze(-1)
        yi_pos_logit = self.item_module(pos_e).squeeze(-1)
        yi_neg_logit = self.item_module(neg_e).squeeze(-1)
        
        yu = self.sigmoid(yu_logit)
        yi_pos = self.sigmoid(yi_pos_logit)
        yi_neg = self.sigmoid(yi_neg_logit)
        
        yui_pos = yk_pos * yu * yi_pos
        yui_neg = yk_neg * yu * yi_neg
        
        loss_o = self.loss(yui_pos, yui_neg)
        
        label_pos = torch.ones_like(yu_logit)
        label_neg = torch.zeros_like(yi_neg_logit)
        
        # # not 100% sure how to translate BPR loss here so instead
        # # we are doing BCE loss with some confident guesses? 
        # #
        # # It might be beneficial to 
        loss_u = self.module_loss(yu_logit, label_pos) 
        loss_i = self.module_loss(yi_pos_logit, label_pos) + self.module_loss(yi_neg_logit, label_neg)
        
        loss = loss_o + self.item_loss_weight * loss_i + self.user_loss_weight * loss_u
        
        return loss        
        