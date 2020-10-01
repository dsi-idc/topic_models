import torch
import torch.nn.functional as F
from torch import nn
import data
import os
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity
import math
import pandas as pd
from googletrans import Translator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ETM(nn.Module):
    def __init__(self, config_dict, machine, embeddings=None):
        super(ETM, self).__init__()
        self.config_dict = config_dict
        self.machine = machine
        # define hyper-parameters
        self.num_topics = config_dict['model_params']['num_topics']
        self.vocab_size = config_dict['vocab_size']
        self.t_hidden_size = config_dict['model_params']['t_hidden_size']
        self.rho_size = config_dict['model_params']['rho_size']
        self.enc_drop = config_dict['optimization_params']['enc_drop']
        self.emsize = config_dict['model_params']['emb_size']
        self.t_drop = nn.Dropout(config_dict['optimization_params']['enc_drop'])
        self.theta_act = self.get_activation(config_dict['model_params']['theta_act'])

        # define the word embedding matrix \rho
        if config_dict['model_params']['train_embeddings']:
            self.rho = nn.Linear(config_dict['model_params']['rho_size'], config_dict['vocab_size'], bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        # define the matrix containing the topic embeddings
        self.alphas = nn.Linear(config_dict['model_params']['rho_size'],
                                config_dict['model_params']['num_topics'], bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))

        # define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(self.vocab_size, self.t_hidden_size),
            self.theta_act,
            nn.Linear(self.t_hidden_size, self.t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        # softmax over vocab dimension
        beta = F.softmax(logit, dim=0).transpose(1, 0)
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        # get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def train_single_epoch(self, epoch, optimizer, train_tokens, train_counts):
        self.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.config_dict['num_docs_train'])
        indices = torch.split(indices, self.config_dict['batch_size'])
        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            self.zero_grad()
            data_batch = data.get_batch(train_tokens, train_counts, ind, self.config_dict['vocab_size'], device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.config_dict['optimization_params']['bow_norm']:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()

            if self.config_dict['optimization_params']['clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config_dict['optimization_params']['clip'])
            optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.config_dict['evaluation_params']['log_interval'] == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        # all_train_loss.append(cur_real_loss)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)
        return cur_real_loss

    def visualize(self, vocab, show_emb=True):
        if not os.path.exists('./results'):
            os.makedirs('./results')
        self.eval()

        # Examples of English queries
        # queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love',
        #            'intelligence', 'money', 'politics', 'health', 'people', 'family']

        queries = ['محمد']

        # visualize topics using monte carlo
        with torch.no_grad():
            print('#'*100)
            print('Visualize topics...')
            topics_words = []
            gammas = self.get_beta()
            for k in range(self.config_dict['model_params']['num_topics']):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.config_dict['evaluation_params']['num_words']+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                print('Topic {}: {}'.format(k, topic_words))

            if show_emb:
                # visualize word embeddings by using V to get nearest neighbors
                print('#'*100)
                print('Visualize word embeddings by using output embedding matrix')
                try:
                    embeddings = self.rho.weight  # Vocab_size x E
                except:
                    embeddings = self.rho         # Vocab_size x E
                neighbors = []
                for word in queries:
                    print('word: {} .. neighbors: {}'.format(
                        word, nearest_neighbors(word, embeddings, vocab)))
                print('#'*100)

    def evaluate(self, source, test_1_tokens, test_1_counts, test_2_tokens, test_2_counts, train_tokens, vocab,
                 tc=False, td=False):
        """Compute perplexity on document completion.
        """
        self.eval()
        with torch.no_grad():
            """
            if source == 'val':
                indices = torch.split(torch.tensor(range(self.config_dict['num_docs_valid'])),
                                      self.config_dict['evaluation_params']['eval_batch_size'])
                tokens = valid_tokens
                counts = valid_counts
            else:
                indices = torch.split(torch.tensor(range(self.config_dict['num_docs_test'])),
                                      self.config_dict['evaluation_params']['eval_batch_size'])
                tokens = test_tokens
                counts = test_counts
            """
            # get \beta here
            beta = self.get_beta()

            # do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(self.config_dict['num_docs_test_1'])),
                                    self.config_dict['evaluation_params']['eval_batch_size'])
            for idx, ind in enumerate(indices_1):
                # get theta from first half of docs
                data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, self.config_dict['vocab_size'], device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if self.config_dict['optimization_params']['bow_norm']:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.get_theta(normalized_data_batch_1)

                # get prediction loss using second half
                data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, self.config_dict['vocab_size'], device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)

                loss = recon_loss / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            if tc or td:
                beta = beta.data.cpu().numpy()
                if tc:
                    print('Computing topic coherence...')
                    get_topic_coherence(beta, train_tokens, vocab)
                if td:
                    print('Computing topic diversity...')
                    get_topic_diversity(beta, 25)
            return ppl_dc

    def fit(self, optimizer, train_tokens, train_counts, test_1_tokens, test_1_counts, test_2_tokens, test_2_counts,
            vocab, ckpt):
        # train model on data
        best_epoch = 0
        best_val_ppl = 1e9
        # defining variables for saving results
        all_val_ppls = []
        all_train_loss = []
        for epoch in range(1, self.config_dict['optimization_params']['epochs']):
            cur_train_loss = self.train_single_epoch(epoch=epoch, optimizer=optimizer, train_tokens=train_tokens,
                                                     train_counts=train_counts)
            all_train_loss.append(cur_train_loss)
            val_ppl = self.evaluate(source='val', test_1_tokens=test_1_tokens, test_1_counts=test_1_counts,
                                    test_2_tokens=test_2_tokens, test_2_counts=test_2_counts,
                                    train_tokens=train_tokens, vocab=vocab, tc=self.config_dict['evaluation_params']['tc'],
                                    td=self.config_dict['evaluation_params']['td'])
            if val_ppl < best_val_ppl:
                with open(ckpt, 'wb') as f:
                    torch.save(self, f)
                best_epoch = epoch
                best_val_ppl = val_ppl
            else:
                # check whether to anneal lr
                lr = optimizer.param_groups[0]['lr']
                if self.config_dict['optimization_params']['anneal_lr'] and \
                        (len(all_val_ppls) > self.config_dict['optimization_params']['nonmono'] and
                         val_ppl > min(all_val_ppls[:-self.config_dict['optimization_params']['nonmono']]) and lr > 1e-5):
                    optimizer.param_groups[0]['lr'] /= self.config_dict['optimization_params']['lr_factor']
            if epoch % self.config_dict['evaluation_params']['visualize_every'] == 0:
                pass
            all_val_ppls.append(val_ppl)
        with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(device)
        val_ppl = model.evaluate(source='val', test_1_tokens=test_1_tokens, test_1_counts=test_1_counts,
                                 test_2_tokens=test_2_tokens, test_2_counts=test_2_counts,
                                 train_tokens=train_tokens, vocab=vocab, tc=self.config_dict['evaluation_params']['tc'],
                                 td=self.config_dict['evaluation_params']['td'])
        results_df = pd.DataFrame({'val_ppls': all_val_ppls, 'train_loss': all_train_loss})
        results_df.to_csv(os.path.join(self.config_dict['saving_models_path'][self.machine], "results_table.csv"))

    def predict_proba(self, bow_tokens):
        """

        :param bow_tokens:
        :return: list of lists
            the distribution over the topics of each document
        """
        theta, kld_theta = self.get_theta(bow_tokens)
        return theta.tolist()

    def predict(self, bow_tokens):
        theta, kld_theta = self.get_theta(bow_tokens)
        # pulling out the index with the maximum value
        res = [cur_theta.index(max(cur_theta)) for cur_theta in theta]
        if all([type(i) is int for i in res]):
            return res
        else:
            raise ValueError("Problem in the Predict function - no single value was found as a max value")

    def print_words_per_topic(self, words_amount, vocab, lang='ar'):
        beta = self.get_beta()
        print('\n')
        if lang == 'ar':
            for k in range(self.num_topics):
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-words_amount + 1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))
        if lang == 'en':
            translator = Translator()
            for k in range(self.num_topics):  # topic_indices:
                gamma = beta[k]
                top_words = \
                    list(gamma.detach().cpu().numpy().argsort()[-words_amount + 1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topic_words_en = list()
                for cur_word in topic_words:
                    result = translator.translate(cur_word, src='ar')
                    topic_words_en.append(result.text)
                print('Topic {}: {}'.format(k, topic_words_en))

    @staticmethod
    def decode(theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    @staticmethod
    def get_activation(act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act
