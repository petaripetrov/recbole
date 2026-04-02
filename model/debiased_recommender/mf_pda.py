from recbole.model.abstract_recommender import PDARecommender
from recbole.model.general_recommender._neumf import _NeuMF


class MF_PDA(PDARecommender, _NeuMF):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)