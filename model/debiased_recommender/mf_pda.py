from recbole.model.abstract_recommender import PDARecommender
from recbole.model.general_recommender._neumf import _NeuMF


class MF_PDA(PDARecommender, _NeuMF):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        score = self.elu(score) + 1

        if self.predict_method == "PDA":
            item_weight = self.propensity_score[item].to(self.device)
            score = score * item_weight

        return score