import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from src.models.lang_model.w2v_averager_model import W2vAveragerModel
from src.models.lang_model.embedding_model import EmbeddingModel
import src.models.time_series.ts_models as ts
import src.models.time_series.latent_to_params as latent_to_params
import numpy as np
import matplotlib.animation as animation



class StripTemporalCF(nn.Module):
    """
    Stripped Megamodel class.

    Comprises:
    --- Always Necessary ---
    Language Model String - Either Avg or Embedding. If avg, uses average
                            word vector features. If embedding, uses a
                            learned embedding from the item id.
    Latent to params hidden dims: A list of the sizes of the hidden units
                                  in the MLP that produces the parameter outputs
    Time model string: Either LinRegress or OnlyMLP. LinRegress learns slope and intercept
                       for each user, question pair. OnlyMLP directly produces
                       the respones. 
    User Latent Model - Embedding module
    Time Model - Matrix Factorization (no time), Linear Regression, Markov Chain
    Latent to Time Series Parameters - MLP for Linear Regression/Markov Chain, Dot Product for Matrix Factorization
    

    --- Optional ---
    Metadata Model - Linear layer (optional), only for politifact task

    ----------------------------------------------------------------------------
    The models are composed in the following structure:

    question_latent_model ----------------\
                                           \
                                            -> latent_to_time_series_parameters -> time_model
                                           /
    user_latent_model --------------------/>

    ----------------------------------------------------------------------------
    """

    def __init__(self, language_model_string,
                 latent_to_params_hidden_dims, time_model_string,
                 task, use_metadata, time_ordinal,
                 language_embed_dim, softmax,
                 metadata_size, num_users,
                 user_embed_size, include_q_embed,
                 question_map_dict, temperature=10):

        """
        Builds the components of the TemporalCF. These are:

        language_model_string (str): language model

        latent_to_params_hidden_dims (list): hidden layer dimension sizes for latent to param
            MLP

        task (str): whether dataset is 'synthetic', 'politifact' or 'fermi'.
            Necessary to use the correct time sequence.

        use_metadata (bool): whether to use metadata in the model

        softmax (bool): whether to use the softmax in the time series model

        time_ordinal: whether the time for the linear regression uses times [0, 1, 2] insted
                      of the actual numerical times.
        
        include_q_embed: Whether to use an item embedding from the question_id
                         as well as the metadata information.

        question_map_dict: Maps question text to id. Necessary for include_q_embed.
         
        temperature: Temperature to use in the softmax.

                """
        super(StripTemporalCF, self).__init__()
        self.time_model_string = time_model_string
        self.include_q_embed = include_q_embed
        ################################
        ######## LANGUAGE MODEL ########
        ################################
        assert language_model_string in ["avg", "embedding"], "language model not implemented"
        if include_q_embed:
            if language_model_string == 'embedding' and question_map_dict is None:
                raise ValueError("if using embeddings, must provide a `question map dict`"
                                 + "to map each question to an embedding")
            elif language_model_string == "avg":
                transform_layer = nn.Linear(50, language_embed_dim)
                self.question_latent_model = nn.Sequential(W2vAveragerModel(False),
                                                           transform_layer)
            elif language_model_string == 'embedding':
                self.question_latent_model = EmbeddingModel(question_map_dict,
                                                            language_embed_dim)
        ################################
        ######## METADATA ########
        ################################
        if use_metadata:
            self.metadata_latent_model = torch.nn.Linear(metadata_size, language_embed_dim)
            self.user_latent_model = torch.nn.Embedding(num_users, user_embed_size)
        else:
            self.metadata_latent_model = None
            self.user_latent_model = torch.nn.Embedding(num_users, user_embed_size)

        ################################
        ######## TIME SERIES MODEL ########
        ################################
        if time_model_string == 'LinRegress':
            if task == 'politifact':
                # times from politifact task
                times = np.array([30, 90, 360])
            else:
                # times from Fermi,
                times = np.array([20, 60, 180])
            if time_ordinal:
                times = np.array([0, 1, 2])
            # convert times to FloatTensor, using cuda if necessary
            self.ts_model = ts.LinRegress(times, temperature=temperature, softmax=softmax)

        elif time_model_string == 'OnlyMLP':
            self.ts_model = ts.OnlyMLP(temperature=temperature, softmax=softmax)
        ################################
        ######## LATENT TO PARAM ########
        ################################
        latent_dimension = user_embed_size
        if use_metadata:
            latent_dimension += language_embed_dim
        if include_q_embed:
            latent_dimension += language_embed_dim

        self.latent_to_param = latent_to_params.CatMLP(latent_dimension,
                                                       latent_to_params_hidden_dims,
                                                       self.ts_model.param_dim)

    def forward(self, user_ids, item_texts, metadata_items, return_vals=False):
        """Combines the forwards of the TemporalCF. Follows structure of class docstring."""
        user_vals = self.user_latent_model.forward(user_ids)

        
        if self.metadata_latent_model is not None:
            meta_vals = self.metadata_latent_model.forward(metadata_items)
            if self.include_q_embed:
                item_vals = self.question_latent_model.forward(item_texts)
                item_vals = torch.cat([item_vals, meta_vals], 1)
            else:
                item_vals = meta_vals
        else:
            if self.include_q_embed:
                item_vals = self.question_latent_model.forward(item_texts)
            else:
                item_vals = None
                
            

        params = self.latent_to_param.forward(
            user_vals, item_vals, None, None)
        ans = self.ts_model.forward(params)
        return ans

    def cuda(self):
        super(StripTemporalCF, self).cuda()
        self.question_latent_model.cuda()
        self.ts_model.cuda()

    def cpu(self):
        super(StripTemporalCF, self).cpu()
        self.question_latent_model.cpu()
        self.ts_model.cpu()


class CompiledStripTemporalCF:
    def __init__(self, network, weight, num_epochs, batch_size,
                 optim_params, use_cuda=False,
                 l1_coef=0, optimizer="SGD", cross_entropy=False):
        """
        fit method takes a ContentDataset and fits it for num_epochs (passed at initialisation)

        Parameters
        ----------

        batch_size (int): the size of each training batch

        network (ContentMF): a network that fits using user_ids and item_texts

        num_epochs (int): the number of training epochs

        optim_params (dict): parameters passed to the Stochastic Gradient Descent (SGD) class

        use_cuda (bool): set to True to use the GPU

        """
        self.batch_size = batch_size
        self.network = network
        self.num_epochs = num_epochs
        self.optim_params = optim_params
        self.loss_fn = nn.MSELoss
        self.use_cuda = use_cuda
        # by default there is no L1 loss, l1_coef = 0
        self.l1_coef = l1_coef
        self.optimizer = optimizer
        self.weight = weight
        self.time_model_string = self.network.time_model_string

        # TODO: probably move use_cuda and dataloader_extract to a Superclass / MixIn
        if use_cuda:
            print('using cuda')
            self.network.cuda()
            self.floattype = torch.cuda.FloatTensor
            self.inttype = torch.cuda.LongTensor
            self.bytetype = torch.cuda.ByteTensor
        else:
            self.floattype = torch.FloatTensor
            self.inttype = torch.LongTensor
            self.bytetype = torch.ByteTensor
        # for param in self.network.parameters():
        #     param = param/100
        #     # TODO: not for language

    def dataloader_extract(self, sample):
        ratings = Variable(sample['rating'].type(self.floattype)).squeeze()
        user_ids = Variable(sample['user_id'].type(self.inttype))
        item_ids = Variable(sample['item_id'].type(self.inttype))
        item_metadata = Variable(sample['item_metadata'].type(self.floattype))
        item_text = sample['item_text']

        return ratings, user_ids, item_ids, item_metadata, item_text

    def weighted_mse_loss(self, inputs, targets, weights):
        return torch.sum(weights * (inputs - targets) ** 2)

    def fit(self, train_set, train_sampler, val_set, val_sampler, verbose,
            patience=5000, eps=1e-4, schedule_epochs=[50],
            schedule_reductions=[5], animate=False):

        assert len(schedule_epochs) == len(schedule_reductions)
        import time
        if animate:
            fig, axes, ims = self.set_up_animation()
        self.network.train()
        t0 = time.time()
        data_loader = DataLoader(
            train_set, batch_size=self.batch_size, sampler=train_sampler)
        param_groups = self.get_param_groups()
        opt = getattr(optim, self.optimizer)(param_groups)
        # for L1 loss
        l1_crit = nn.L1Loss(size_average=False)
        train_loss_list = []
        mse_val_loss_list = []
        time_list = []
        stopping_counter = 0
        min_val_loss = None

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            total_scored = 0.0  # Number of ratings we score on

            # Schedule the reduction in the learning rates
            if len(schedule_epochs) > 0:
                if epoch == schedule_epochs[0]:
                    schedule_epochs.pop(0)
                    reduction = schedule_reductions.pop(0)
                    for p in opt.param_groups:
                        p['lr'] = p['lr'] / reduction

            for i, sample in enumerate(data_loader):
                ratings, user_ids, item_ids, item_metadata, item_text = self.dataloader_extract(
                    sample)

                # We can get some wacky results if we feed in batches
                # that are not full-size, so check for this
                if ratings.size == torch.Size([]):
                    continue  # If rating is a singleton tensor
                if len(ratings) < self.batch_size:
                    continue

                opt.zero_grad()
                # We form the prediction as a bs x 3 tensor.
                # We don't have all the actual responses, so need to
                # filter out the corresponding predictions where there
                # are nans in the responses
                pred = self.network.forward(user_ids, item_text, item_metadata)
                ones = torch.Tensor(torch.ones(ratings.shape))
                weight_matrix = (torch.Tensor(self.weight)*ones).type(self.floattype)

                # Calculate response mask from ratings
                response_mask = ~np.isnan(ratings)
                response_mask = response_mask.type(self.bytetype)

                # Now we need to only update on the provided targets
                masked_ratings = torch.masked_select(ratings, response_mask)
                masked_weights = torch.masked_select(weight_matrix, response_mask)
                masked_preds = torch.masked_select(pred, response_mask)
                loss = self.weighted_mse_loss(masked_preds, masked_ratings, masked_weights)

                # L1 loss
                reg_loss = 0
                for name, param in self.network.named_parameters():
                    reg_loss += l1_crit(param, torch.zeros_like(param))
                loss += self.l1_coef * reg_loss

                loss.backward()
                opt.step()

                epoch_loss += loss.data
                total_scored += len(masked_ratings)

            tdiff = time.time() - t0
            assert total_scored > 0, "Train set is smaller than batch_size"
            av_train_loss = (
                epoch_loss / total_scored)
            val_outs = self.validate(val_set, val_sampler)
            val_mse_loss, masked_preds, masked_ratings, masked_weights, _ = val_outs
            av_val_loss = val_mse_loss
            if animate:
                fig, axes, ims = self.add_animation_panels(fig, axes, ims,
                                                           train_set,
                                                           train_sampler,
                                                           masked_ratings,
                                                           masked_preds,
                                                           response_mask,
                                                           pred, train_loss_list,
                                                           epoch, mse_val_loss_list)

            # Early stopping
            if min_val_loss is None:
                min_val_loss = av_val_loss

            elif av_val_loss < min_val_loss - eps:
                # Reset counter if we have improved validation loss
                min_val_loss = av_val_loss
                stopping_counter = 0
            else:
                # Increment counter, stopping if we have plateaued
                stopping_counter += 1
                if stopping_counter > patience:
                    print('Stopping at epoch {}'.format(epoch))
                    break

            if verbose:
                print('epoch {0:4d}\ttrain_loss = {1:6.5f}\tval_mse_loss = {2:6.5f}\tElapsed time: {3:8.1f}'.format(
                    epoch, av_train_loss, val_mse_loss, tdiff))
            train_loss_list.append(av_train_loss.item())
            mse_val_loss_list.append(val_mse_loss.item())
            time_list.append(tdiff)
        if animate:
            ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                            repeat_delay=1000, repeat=True)
            import time
            ani.save('../results/test{}.mp4'.format(str(time.time()).split('.')[0]), fps=15)

        return train_loss_list, mse_val_loss_list, time_list

    def _extract_data(self, dataset, sampler):
        data_loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)
        assert len(data_loader) == 1, 'data loader should have size 1'
        sample = next(data_loader.__iter__()) # dataloader takes one (sub)set of the dataset
        ratings, user_ids, item_ids, item_metadata, item_text = self.dataloader_extract(sample)
        assert ratings.size() != torch.Size([]), 'ratings size empty'
        assert len(ratings) >= self.batch_size, 'not enough ratings compared to batch size'
        return ratings, user_ids, item_ids, item_metadata, item_text

    def _predict(self, user_ids, item_text, item_metadata):
        self.network.eval()
        pred = self.network.forward(user_ids, item_text, item_metadata)
        self.network.train()
        return pred

    def predict(self, dataset, sampler):
        ratings, user_ids, item_ids, item_metadata, item_text = self._extract_data(dataset, sampler)
        pred = self._predict(user_ids, item_text, item_metadata)
        return pred.detach().numpy()

    def validate(self, dataset, sampler):
        """For validate we assume we are looking at a set of responses with
        only slow entries. We set the weight matrix to the unit so that
        the validation mse is the actual MSE (as opposed to the train,
        which can be different due to the weighting of different times)"""
        ratings, user_ids, item_ids, item_metadata, item_text = self._extract_data(dataset, sampler)
        # valid_loss = self.loss_fn()
        pred = self._predict(user_ids, item_text, item_metadata)

        weight_matrix = torch.Tensor(torch.ones(ratings.shape)).type(self.floattype)

        response_mask = ~np.isnan(ratings)
        response_mask = response_mask.type(self.bytetype)
        masked_ratings = torch.masked_select(ratings, response_mask)
        masked_preds = torch.masked_select(pred, response_mask)
        masked_weights = torch.masked_select(weight_matrix, response_mask)
        val_mse_loss = self.weighted_mse_loss(masked_preds, masked_ratings, masked_weights)
        av_val_mse_loss = val_mse_loss.data / len(masked_ratings)

        # Return item list sorted by MSE on last item
        mse = ((pred - ratings)**2)
        zipped_responses = zip(mse.cpu().data.numpy(),
                               masked_preds.cpu().data.numpy(),
                               masked_ratings.cpu().data.numpy(),
                               user_ids.cpu().data.numpy(),
                               item_ids.cpu().data.numpy(),
                               item_text,
                               item_metadata)

        return av_val_mse_loss, masked_preds, masked_ratings, masked_weights, list(zipped_responses)

    def set_up_animation(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 5, figsize=(40, 20))
        axes[0, 0].set_xlim([-0.1, 1.1])
        axes[0, 0].set_ylim([-0.1, 1.1])
        axes[0, 1].set_xlim([-0.1, 1.1])
        axes[0, 1].set_ylim([-0.1, 1.1])
        axes[1, 4].set_ylim([0.09, 0.13])
        ims = []
        return fig, axes, ims

    def get_param_groups(self, remove_lang_weight_decay=True):
        """Returns a list of parameter groups dictionaries.
        If remove_lang_weight_decay, we stop any weight decay
        happening to the language model as it's probably not a good
        idea to regularize the LSTM weights directly in that way."""

        param_groups_list = []
        for sub_model in self.network.children():
            params = filter(lambda p: p.requires_grad,
                            sub_model.parameters())
            optim_params = self.optim_params.copy()
            optim_params['params'] = params
            param_groups_list.append(optim_params)

        return param_groups_list

    def add_animation_panels(self, fig, axes, ims, train_set,
                             train_sampler, masked_ratings,
                             masked_preds, response_mask, pred,
                             train_loss_list, epoch, mse_val_loss_list):
        _, _, train_masked_preds, train_masked_ratings, _, _ = self.validate(train_set, train_sampler)
        im1 = axes[0, 0].scatter(train_masked_ratings.data, train_masked_preds.data, c='b', alpha=0.1)
        im2 = axes[0, 1].scatter(masked_ratings.data, masked_preds.data, c='b', alpha=0.5)

        metadata_weights = self.network.metadata_latent_model.weight.detach().numpy()[0]
        im3 = axes[0, 2].hist(metadata_weights, 20, color='C1', edgecolor='k', linewidth=1)
        axes[0, 2].set_xlabel('Metadata Weights')

        user_weights = self.network.user_latent_model.weight.detach().numpy()
        im4 = axes[0, 3].hist(user_weights, np.linspace(-2, 2, 21), color='C2', linewidth=1, edgecolor='k')
        axes[0, 3].set_xlabel('User Weights')

        latent_to_param_weights_1 = self.network.latent_to_param.net.layers[0].weight.detach().numpy().flatten()
        latent_to_param_weights_2 = self.network.latent_to_param.net.layers[1].weight.detach().numpy().flatten()
        latent_to_param_weights_3_0 = self.network.latent_to_param.net.layers[2].weight[0, :].detach().numpy().flatten()
        latent_to_param_weights_3_1 = self.network.latent_to_param.net.layers[2].weight[1, :].detach().numpy().flatten()

        im5 = axes[0, 4].hist(latent_to_param_weights_1, 10, color='C3', linewidth=1, edgecolor='k')
        axes[0, 4].set_xlabel('Latent to Param Weights Layer 1')

        im6 = axes[1, 0].hist(latent_to_param_weights_2, 10, color='C3', linewidth=1, edgecolor='k')
        axes[1, 0].set_xlabel('Latent to Param Weights Layer 2')

        im7 = axes[1, 1].hist(latent_to_param_weights_3_0, 10, color='C3', linewidth=1, edgecolor='k')
        axes[1, 1].set_xlabel('Latent to Param Weights Layer 3 to Slope')

        im8 = axes[1, 2].hist(latent_to_param_weights_3_1, 10, color='C3', linewidth=1, edgecolor='k')
        axes[1, 2].set_xlabel('Latent to Param Weights Layer 3 to Intercept')

        im9 = []
        for index, response_row in enumerate(response_mask):
            if response_row[0] == 1 and response_row[1] == 1 and response_row[2] == 1:
                # Only plot for full rows:
                line = axes[1, 3].plot([0, 1, 2], pred[index, :].detach().numpy(),
                                       c='k', alpha=0.2)
                im9.append(line[0])

        im10 = []
        if len(train_loss_list) > 0:
            line2 = axes[1, 4].plot(range(epoch), train_loss_list, c='b')
            line20 = axes[1, 4].scatter([epoch-1], train_loss_list[-1], c='b', s=15)
            line3 = axes[1, 4].plot(range(epoch), mse_val_loss_list, c='r')
            line30 = axes[1, 4].scatter([epoch-1], mse_val_loss_list[-1], c='r', s=15)

            im10.extend([line2[0], line3[0], line20, line30])

        to_list = [im1, im2]
        to_list.extend(im3[2])
        to_list.extend(im4[2])
        to_list.extend(im5[2])
        to_list.extend(im6[2])
        to_list.extend(im7[2])
        to_list.extend(im8[2])
        to_list.extend(im9)
        to_list.extend(im10)
        ims.append(to_list)

        return fig, axes, ims
