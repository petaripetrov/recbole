import torch
from torch import nn

from recbole.model.abstract_recommender import AbstractRecommender
from recbole.model.loss import EmbLoss
from recbole.utils.enum_type import InputType, ModelType
from tqdm import tqdm


class RegLossToOne(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self, l2=0.01):
        super(RegLossToOne, self).__init__()

        self.l2 = l2

    def forward(self, parameters: torch.Tensor):
        return self.l2 * torch.sum(torch.square(parameters - 1))

class Filter(nn.Module):
    def __init__(self, input_dim, filter_units = [128, 128, 128, 128, 128], reg_term=0.0001):
        super(Filter, self).__init__()
        self.filter_units = filter_units
        self.reg_term = reg_term
        self.mlp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.film_alpha_layers = nn.ModuleList()
        self.film_beta_layers = nn.ModuleList()

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.reg_loss_one = RegLossToOne()
        self.reg_loss = EmbLoss()

        input_size = input_dim
        for i, unit in enumerate(self.filter_units):
            self.mlp_layers.append(
                nn.Linear(input_size, unit)
            )
            self.film_alpha_layers.append(
                nn.Linear(input_dim, unit, bias=False)
            )
            self.film_beta_layers.append(
                nn.Linear(input_dim, unit, bias=False)
            )
            
            self.bn_layers.append(nn.BatchNorm1d(unit))

            input_size = unit

    def forward(self, inputs, training=True):
        embedding = inputs
        self.alpha_reg_loss = 0.0
        self.beta_reg_loss = 0.0

        for i, unit in enumerate(self.filter_units):
            filter_layer = self.mlp_layers[i]
            film_alpha_layer = self.film_alpha_layers[i]
            film_beta_layer = self.film_beta_layers[i]
            bn_layer = self.bn_layers[i]

            embedding = filter_layer(embedding)
            embedding = self.relu(embedding)
            alpha = film_alpha_layer(inputs)
            beta = film_beta_layer(inputs)

            self.alpha_reg_loss += self.reg_loss_one(alpha)
            self.beta_reg_loss += self.reg_term * self.reg_loss(beta)

            embedding = embedding * alpha + beta

            if training:
                bn_layer.train()
            else:
                bn_layer.eval()
            embedding = bn_layer(embedding)
            embedding = self.leaky_relu(embedding)

        return embedding
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, disc_type, reg_term=0.0001, implicit_layer_units = [128, 64, 32, 16, 8], explicit_layer_units = [128, 64, 32, 16, 8]):
        super(Discriminator, self).__init__()
        self.reg_term = reg_term

        if disc_type == "implicit":
            self.descriminator_units = implicit_layer_units
        elif disc_type == "explicit":
            self.descriminator_units = explicit_layer_units
        else:
            raise Exception(self.descriminator_units)
        
        self.mlp_layers = nn.ModuleList()
        
        input_size = input_dim
        for i, unit in enumerate(self.descriminator_units):
            self.mlp_layers.append(
                nn.Linear(input_size, unit)
            )
            input_size = unit

        self.pred_layer = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()
        self.emb_loss = EmbLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embedding = inputs
        self.reg_loss = 0.0

        for i, unit in enumerate(self.descriminator_units):
            descriminator_layer = self.mlp_layers[i]
            embedding = descriminator_layer(embedding)
            embedding = self.relu(embedding)

            # Might be better to do this directly on the weights outside of the loop
            self.reg_loss += self.reg_term * self.emb_loss(embedding)

        pred: torch.Tensor = self.pred_layer(embedding)
        pred = self.sigmoid(pred)
        return pred.squeeze(-1)
    
class BaseRecModel(nn.Module): # BPR equivalent?
    def __init__(self, input_dim, reg_term=0.0001, rec_layer_units = [128, 64, 32, 16, 8]):
        super(BaseRecModel, self).__init__()
        self.reg_term = reg_term 
        self.base_units = rec_layer_units
        self.mlp_layers = nn.ModuleList()

        self.relu = nn.ReLU()
        self.reg_loss = EmbLoss()

        input_size = input_dim
        for i, unit in enumerate(self.base_units):
            self.mlp_layers.append(nn.Linear(input_size, unit))
            input_size = unit
        
        self.pred_layer = nn.Linear(input_size, 1)

    def forward(self, inputs):
        embedding = inputs

        for i, unit in enumerate(self.base_units):
            rec_layer = self.mlp_layers[i]
            embedding = rec_layer(embedding)
            embedding = self.relu(embedding)
        
        pred = self.pred_layer(embedding)
        pred = self.relu(pred)

        return pred.squeeze()
    

class FAiR(AbstractRecommender):
    type = ModelType.GENERAL

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, d_step=5, g_step=1, reg_term=0.0001, n_user_samples=200, n_item_samples=128, lambda1 = 1.0, lambda2 = 1.0, lambda3 = 1.0, dimension=128):
        super(FAiR, self).__init__()

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.PROPENSITIES = config["PROPENSITY_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.RATING = config['RATING_FIELD']
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.config = config
        self.d_step = d_step
        self.g_step = g_step
        self.reg_term = reg_term
        self.n_user_samples = n_user_samples
        self.n_item_samples = n_item_samples
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3

        user_group = torch.zeros(self.n_users)
        item_group = torch.zeros(self.n_items)
        # avg_rating_df = pd.DataFrame(df.groupby('item')['rating'].mean())
        avg_rating = dataset.avg_rating
        self.register_buffer('user_group', torch.tensor(user_group.clone().detach().long().squeeze(), dtype=torch.int64).squeeze())
        self.register_buffer('item_group', torch.tensor(item_group.clone().detach().long().squeeze(), dtype=torch.int64).squeeze())
        self.register_buffer('average_rating', torch.tensor(avg_rating.clone().detach().long().squeeze(), dtype=torch.int64).squeeze())

        self.user_embedding_layer = nn.Embedding(self.n_users, dimension)
        self.item_embedding_layer = nn.Embedding(self.n_items, dimension)

        self.user_filter = Filter(dimension)
        self.item_filter = Filter(dimension)

        filter_out_dim = 128 # hardcoded but should be taken froom config
        
        self.user_explicit_discriminator = Discriminator(filter_out_dim, "explicit")
        self.item_explicit_discriminator = Discriminator(filter_out_dim, "explicit")
        self.implicit_discriminator = Discriminator(n_item_samples, "implicit")

        self.rec_model = BaseRecModel(filter_out_dim * 2)
        
        self.pretrain_loss = nn.MSELoss()
        self.user_d_loss = nn.BCELoss()
        self.item_d_loss = nn.BCELoss()
        self.im_d_loss = nn.BCELoss()
        self.rec_loss = nn.MSELoss() # torch.F.mse_loss()

        self.pretrain_optimizer = torch.optim.Adam(
            list(self.user_embedding_layer.parameters()) +
            list(self.item_embedding_layer.parameters())
        )
        self.d_optimizer = torch.optim.Adam(
            list(self.user_explicit_discriminator.parameters()) +
            list(self.item_explicit_discriminator.parameters()) +
            list(self.implicit_discriminator.parameters())
        )
        self.g_optimizer = torch.optim.Adam(
            list(self.user_filter.parameters()) +
            list(self.item_filter.parameters()) +
            list(self.rec_model.parameters())
        )

        self.is_pretrained = False

    def forward(self, user, item, training=True):
        self.user_filter.train(training)
        self.item_filter.train(training)

        user_emb = self.user_embedding_layer(user)
        item_emb = self.item_embedding_layer(item)
        user_emb = self.user_filter(user_emb, training)
        item_emb = self.item_filter(item_emb, training)
        joint_emb = torch.cat([user_emb, item_emb], dim=-1)
        result = self.rec_model(joint_emb)

        return result.squeeze()
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item, training=False)
    
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        rating = interaction[self.RATING]

        if not self.is_pretrained:
            return self._pretrain(user, item, rating)
        else:
            return self._adv_train(user, item, rating)
        
    def _pretrain(self, user, item, rating):
        user_emb = self.user_embedding_layer(user)
        item_emb = self.item_embedding_layer(item)
        out = torch.sum(user_emb * item_emb, dim=1).squeeze()
        loss = self.pretrain_loss(out, rating)

        loss.backward()
        self.pretrain_optimizer.step()

        return loss
    
    def _adv_train(self, user, item, y_true):
        device = user.device

        # --- Sample random users/items ---
        # These embeddings are computed OUTSIDE the discriminator gradient tape
        # in the original, so we use no_grad here
        user_g = torch.randint(0, self.n_users, (self.n_users // 10,), device=device)
        item_g = torch.randint(0, self.n_items, (self.n_items // 10,), device=device)

        user_group_true = self.user_group[user_g].float()
        item_group_true = self.item_group[item_g].float()

        with torch.no_grad():
            user_emb_d = self.user_filter(self.user_embedding_layer(user_g), training=False)
            item_emb_d = self.item_filter(self.item_embedding_layer(item_g), training=False)

        # --- Compute ru/ro vectors outside discriminator tape ---
        # In the TF original these are computed before the d_step loop,
        # outside the GradientTape, so gradients do not flow through them
        sampled_user_ids = torch.randint(0, self.n_users, (self.n_user_samples,), device=device)
        sampled_item_ids = torch.randint(0, self.n_items, (self.n_item_samples,), device=device)
        sampled_users = sampled_user_ids.repeat_interleave(self.n_item_samples)
        sampled_items = sampled_item_ids.repeat(self.n_user_samples)

        with torch.no_grad():
            ru_vectors = self.forward(sampled_users, sampled_items, training=False)
            ru_vectors = ru_vectors.reshape(self.n_user_samples, self.n_item_samples)
            ro_vectors = self.average_rating[sampled_item_ids].reshape(1, self.n_item_samples)

        ru_labels = torch.zeros(self.n_user_samples, device=device)
        ro_labels = torch.ones(1, device=device)

        # --- Discriminator steps (d_optimizer only) ---
        for _ in range(self.d_step):
            self.d_optimizer.zero_grad()

            user_group_pred = self.user_explicit_discriminator(user_emb_d)
            item_group_pred = self.item_explicit_discriminator(item_emb_d)
            ru_pred = self.implicit_discriminator(ru_vectors)
            ro_pred = self.implicit_discriminator(ro_vectors)

            d_loss = (
                self.user_d_loss(user_group_pred, user_group_true)
                + self.item_d_loss(item_group_pred, item_group_true)
                + self.im_d_loss(ru_pred.reshape(ru_labels.shape), ru_labels)
                # ro_pred needs unsqueeze to match ro_labels shape,
                # equivalent to tf.expand_dims in the original
                + self.im_d_loss(ro_pred.reshape(ro_labels.shape), ro_labels)
                + self.user_explicit_discriminator.reg_loss
                + self.item_explicit_discriminator.reg_loss
                + self.implicit_discriminator.reg_loss
            )

            d_loss.backward()
            self.d_optimizer.step()

        # --- Generator steps (g_optimizer only) ---
        # user/item group targets are FLIPPED here (abs(group - 1))
        # this is the adversarial part: generator tries to fool discriminator
        user_group_true = torch.abs(self.user_group[user].float() - 1)
        item_group_true = torch.abs(self.item_group[item].float() - 1)

        # For ru_vectors in the generator phase, we use the actual batch users
        # (not random samples) crossed with random items
        sampled_user_ids = user.squeeze()
        n_sampled_user = sampled_user_ids.shape[0]
        sampled_item_ids = torch.randint(0, self.n_items, (self.n_item_samples,), device=device)
        sampled_users = sampled_user_ids.repeat_interleave(self.n_item_samples)
        sampled_items = sampled_item_ids.repeat(n_sampled_user)

        # ru_labels are ONES here (opposite of discriminator phase)
        # generator wants the discriminator to predict 1 (real) for its outputs
        ru_labels = torch.ones(n_sampled_user, device=device)

        for _ in range(self.g_step):
            self.g_optimizer.zero_grad()

            # These are recomputed inside the g_step loop with training=True
            # so BatchNorm and gradients behave correctly
            user_emb = self.user_filter(self.user_embedding_layer(user), training=True)
            item_emb = self.item_filter(self.item_embedding_layer(item), training=True)

            rec_pred = self.forward(user, item, training=True)
            
            with torch.no_grad():
                ru_vectors = self.forward(sampled_users, sampled_items, training=False)
                ru_vectors = ru_vectors.reshape(n_sampled_user, self.n_item_samples)
                
            ru_pred = self.implicit_discriminator(ru_vectors)
            user_group_pred = self.user_explicit_discriminator(user_emb)
            item_group_pred = self.item_explicit_discriminator(item_emb)

            g_loss = (
                self.rec_loss(rec_pred, y_true)
                + self.l1 * self.user_d_loss(user_group_pred, user_group_true)
                + self.l2 * self.item_d_loss(item_group_pred, item_group_true)
                + self.l3 * self.im_d_loss(ru_pred.reshape(ru_labels.shape), ru_labels)
                # Filter losses from both user and item filters
                # equivalent to tf.add_n(self.user_filter.losses) in original
                + self.user_filter.alpha_reg_loss + self.user_filter.beta_reg_loss
                + self.item_filter.alpha_reg_loss + self.item_filter.beta_reg_loss
            )

            g_loss.backward()
            self.g_optimizer.step()

        return g_loss

# class FAiRRecommender(AbstractRecommender):

