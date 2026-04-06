from recbole.model.abstract_recommender import PCCRecommender
from recbole.model.general_recommender._mf import _MF
from recbole.model.init import xavier_normal_initialization


class MF_PCC(PCCRecommender, _MF):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.apply(xavier_normal_initialization)
        
    def _calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        loss = self.loss(output, label)
        
        return loss, output