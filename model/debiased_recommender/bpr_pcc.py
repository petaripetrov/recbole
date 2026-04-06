import torch

from recbole.model.abstract_recommender import PCCRecommender
from recbole.model.general_recommender._bpr import _BPR
from recbole.model.init import xavier_normal_initialization

class BPR_PCC(PCCRecommender, _BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.apply(xavier_normal_initialization)
    
    def _calculate_loss(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        # Same as _BPR except we also return pos_item_score
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        
        loss = self.loss(pos_item_score, neg_item_score)
        
        return loss, pos_item_score