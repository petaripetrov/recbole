import torch

from recbole.model.abstract_recommender import PDARecommender
from recbole.model.general_recommender._lightgcn import _LightGCN


class LightGCN_PDA(PDARecommender, _LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward()

    def calculate_loss(self, interaction):
        # Example implementation based on BPR
        # TLDR: scale scores wth Elu and then with propensities

        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embedding = item_all_embeddings[pos_item]
        neg_embedding = item_all_embeddings[neg_item]

        pos_item_weight = self.propensity_score[pos_item].to(self.device)
        neg_item_weight = self.propensity_score[neg_item].to(self.device)
        
        pos_score = self.elu(torch.mul(u_embeddings, pos_embedding).sum(dim=1)) + 1
        pos_score = pos_score * pos_item_weight
        neg_score = self.elu(torch.mul(u_embeddings, neg_embedding).sum(dim=1)) + 1
        
        neg_score = neg_score * neg_item_weight
        mf_loss = self.mf_loss(pos_score, neg_score)

        u_ego_embeddings = self.user_embedding(user)
        p_ego_embeddings = self.item_embedding(pos_item)
        n_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            p_ego_embeddings,
            n_ego_embeddings,
            require_pow=self.require_pow
        )

        loss = mf_loss + self.reg_weight * reg_loss
        return loss
    

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = self.elu(torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))) + 1

        if self.predict_method == 'PDA':
            item_weight = self.propensity_score.to(self.device)
            scores = scores * item_weight

        return scores.view(-1)
