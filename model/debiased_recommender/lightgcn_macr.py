import torch

from recbole.model.abstract_recommender import MACRRecommender
from recbole.model.general_recommender._lightgcn import _LightGCN
from recbole.utils.enum_type import InputType


class LightGCN_MACR(MACRRecommender, _LightGCN):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):

        super().__init__(config, dataset)

    def _forward(self, user, item):
        return super()._forward(user, item)
    
    def calculate_loss(self, interaction):
        # Abstract away
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_e, item_all_e = self._forward(user, item)
        u_e = user_all_e[user]
        pos_e = item_all_e[item]
        neg_e = item_all_e[neg_item]

        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)
        yu = self.sigmoid(self.user_module(user_all_e))[user].squeeze(-1)
        yi = self.sigmoid(self.item_module(item_all_e)).squeeze(-1)
        yi_pos = yi[item]
        yi_neg = yi[neg_item]
        
        yui_pos = pos_scores * self.sigmoid(yu) * self.sigmoid(yi_pos)
        yui_neg = neg_scores * self.sigmoid(yu) * self.sigmoid(yi_neg)
        
        mf_loss = self.mf_loss(yui_pos, yui_neg)
        
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        # Use LightGCN loss 
        label = torch.ones(item_all_e.shape[0])
        label[neg_item] = 0
        label = label.to(self.device)

        loss_o = mf_loss + self.reg_weight * reg_loss
        loss_i = self.module_loss(yi_pos, torch.ones_like(yi_pos)) + self.module_loss(yi_neg, torch.ones_like(yi_neg))
        loss_u = self.module_loss(yu, torch.ones_like(yu))

        return loss_o + self.item_loss_weight * loss_i + self.user_loss_weight * loss_u 




