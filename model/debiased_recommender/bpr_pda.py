import torch

from recbole.model.abstract_recommender import PDARecommender
from recbole.model.general_recommender._bpr import _BPR
from recbole.model.init import xavier_normal_initialization


class BPR_PDA(PDARecommender, _BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.apply(xavier_normal_initialization)
    
    def forward(self, user, item):
        return super().forward(user, item)
    
    def calculate_loss(self, interaction):
        return super()._calculate_loss(interaction)
        
    def forward(self, user, item) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(user, item)

    def calculate_loss(self, interaction):
        # Example implementation based on BPR
        # TLDR: scale scores wth Elu and then with propensities
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_item_weight = self.propensity_score[pos_item].to(self.device)
        neg_item_weight = self.propensity_score[neg_item].to(self.device)

        pos_score = self.elu(torch.mul(user_e, pos_e).sum(dim=1)) + 1
        pos_score = pos_score * pos_item_weight
        neg_score = self.elu(torch.mul(user_e, neg_e).sum(dim=1)) + 1
        neg_score = neg_score * neg_item_weight

        loss = self.loss(pos_score, neg_score)
        reg_loss = self.reg_weight * self.reg_loss(user_e, pos_e, neg_e)
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = self.elu(torch.matmul(user_e, all_item_e.transpose(0, 1))) + 1  # [user_batch_num,item_tot_num]
        if self.predict_method == 'PDA':
            item_weight = self.propensity_score.to(self.device)
            score = score * item_weight
        return score.view(-1)