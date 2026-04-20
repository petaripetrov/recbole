import torch

from recbole.model.abstract_recommender import PDARecommender
from recbole.model.general_recommender._mf import _MF


class MF_PDA(PDARecommender, _MF):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
    def forward(self, user, item):
        return super().forward(user, item)
    
    def calculate_loss(self, interaction):
        return super()._calculate_loss(interaction)
        
    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1), user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        
        weight = self.propensity_score[interaction[self.column]].to(self.device)
        
        score, user_e, item_e = self.forward(user, item)
        
        score = self.elu(score) + 1
        score *= weight
        
        loss = self.loss(score, label)
        reg_loss = self.reg_weight * self.reg_loss(user_e, item_e)
        
        return loss + reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        score = self.elu(score) + 1

        if self.predict_method == "PDA":
            item_weight = self.propensity_score[interaction[self.column]].to(self.device)
            score = score * item_weight

        return score