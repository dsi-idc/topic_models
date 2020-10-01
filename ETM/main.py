# Author: Avrahami (abraham.israeli@post.idc.ac.il), last update: 1.10.2020
#/usr/bin/python
from __future__ import print_function
import data
import torch
import numpy as np 
import os
from torch import optim
from etm import ETM
from utils import prepare_embedding_matrix
import commentjson
from data_prep.arabic_twitter import ArabicTwitterPreProcess

machine = 'AVRAHAMI-PC'


def _set_optimizer():
    if config_dict['optimization_params']['optimizer'] == 'adam':
        selected_optimizer = optim.Adam(etm_model.parameters(), lr=config_dict['optimization_params']['lr'],
                                        weight_decay=config_dict['optimization_params']['wdecay'])
    elif config_dict['optimization_params']['optimizer'] == 'adagrad':
        selected_optimizer = optim.Adagrad(etm_model.parameters(), lr=config_dict['optimization_params']['lr'],
                                           weight_decay=config_dict['optimization_params']['wdecay'])
    elif config_dict['optimization_params']['optimizer'] == 'adadelta':
        selected_optimizer = optim.Adadelta(etm_model.parameters(), lr=config_dict['optimization_params']['lr'],
                                            weight_decay=config_dict['optimization_params']['wdecay'])
    elif config_dict['optimization_params']['optimizer'] == 'rmsprop':
        selected_optimizer = optim.RMSprop(etm_model.parameters(), lr=config_dict['optimization_params']['lr'],
                                           weight_decay=config_dict['optimization_params']['wdecay'])
    elif config_dict['optimization_params']['optimizer'] == 'asgd':
        selected_optimizer = optim.ASGD(etm_model.parameters(), lr=config_dict['optimization_params']['lr'],
                                        t0=0, lambd=0., weight_decay=config_dict['optimization_params']['wdecay'])
    else:
        print('Defaulting to vanilla SGD')
        selected_optimizer = optim.SGD(etm_model.parameters(), lr=config_dict['optimization_params']['lr'])
    return selected_optimizer


if __name__ == "__main__":
    # loading the config file
    config_dict = commentjson.load(open('config.json'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(config_dict['random_seed'])
    torch.manual_seed(config_dict['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config_dict['random_seed'])

    # data prep (replaces the data_intuview_arabic.py file process) - files will be saved in the required location
    if eval(config_dict['prepare_data']):
        preprocess_obj = ArabicTwitterPreProcess(config_dict=config_dict, machine=machine)
        #preprocess_obj._calculate_stats(data_path=config_dict['raw_data_path'][machine])
        preprocess_obj.fit_transform(data_path=config_dict['raw_data_path'][machine], verbose=True)
        if eval(config_dict['data_prep_params']['save_model']):
            preprocess_model_f_name = config_dict['data_prep_params']['saving_model_f_name'] + '.p'
            preprocess_obj.save_obj(f_name=preprocess_model_f_name)
    # get data
    # 1. vocabulary
    vocab, train, valid, test = data.get_data(os.path.join(config_dict['data_path'][machine]))
    vocab_size = len(vocab)
    config_dict['vocab_size'] = vocab_size

    # 1. training data
    train_tokens = train['tokens']
    train_counts = train['counts']
    config_dict['num_docs_train'] = len(train_tokens)

    # 2. dev set
    valid_tokens = valid['tokens']
    valid_counts = valid['counts']
    config_dict['num_docs_valid'] = len(valid_tokens)

    # 3. test data
    test_tokens = test['tokens']
    test_counts = test['counts']
    config_dict['num_docs_test'] = len(test_tokens)
    test_1_tokens = test['tokens_1']
    test_1_counts = test['counts_1']
    config_dict['num_docs_test_1'] = len(test_1_tokens)
    test_2_tokens = test['tokens_2']
    test_2_counts = test['counts_2']
    config_dict['num_docs_test_2'] = len(test_2_tokens)

    embeddings = None
    # in case we gave as input the embeddings file to be used as pre-trained model
    if not eval(config_dict['model_params']['train_embeddings']):
        embeddings = prepare_embedding_matrix(emb_data_path=config_dict['emb_path'][machine],
                                              emb_size=config_dict['model_params']['emb_size'], vocab=vocab,
                                              random_seed=config_dict['random_seed'])
        # updating the required values after the function returned the embbeddings
        embeddings = torch.from_numpy(embeddings).to(device)
        config_dict['embeddings_dim'] = embeddings.size()
    print('=*'*100)
    print('Training an Embedded Topic Model on {} '
          'with the following settings: {}'.format(config_dict['dataset'].upper(), config_dict))
    print('=*'*100)

    # define checkpoint
    if not os.path.exists(config_dict['saving_models_path'][machine]):
        os.makedirs(config_dict['saving_models_path'][machine])

    if config_dict['optimization_params']['mode'] == 'eval':
        ckpt = config_dict['evaluation_params']['load_from']
    else:
        ckpt = \
            os.path.join(config_dict['saving_models_path'][machine],
                         'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_'
                         'trainEmbeddings_{}'.format(config_dict['dataset'], config_dict['model_params']['num_topics'],
                                                     config_dict['model_params']['t_hidden_size'],
                                                     config_dict['optimization_params']['optimizer'],
                                                     config_dict['optimization_params']['clip'],
                                                     config_dict['model_params']['theta_act'],
                                                     config_dict['optimization_params']['lr'],
                                                     config_dict['batch_size'],
                                                     config_dict['model_params']['rho_size'],
                                                     config_dict['model_params']['train_embeddings']))

    # define model and optimizer
    etm_model = ETM(config_dict=config_dict, machine=machine, embeddings=embeddings)
    print('model: {}'.format(etm_model))
    optimizer = _set_optimizer()

    if config_dict['optimization_params']['mode'] == 'train':
        etm_model.fit(optimizer=optimizer, train_tokens=train_tokens, train_counts=train_counts,
                      test_1_tokens=test_1_tokens, test_1_counts=test_1_counts, test_2_tokens=test_2_tokens,
                      test_2_counts=test_2_counts, vocab=vocab, ckpt=ckpt)

    elif config_dict['optimization_params']['mode'] == 'predict':
        preprocess_model_f_name = config_dict['data_prep_params']['saving_model_f_name'] + '.p'
        preprocess_obj = ArabicTwitterPreProcess.load_obj(f_path=config_dict['saving_models_path'][machine],
                                                          f_name=preprocess_model_f_name)
        bow_new_docs_tokens, bow_new_docs_counts = preprocess_obj.transform(data_path=config_dict['raw_data_path'][machine])
        data_batch = data.get_batch(bow_new_docs_tokens, bow_new_docs_counts,
                                    torch.tensor(list(range(len(bow_new_docs_tokens)))), vocab_size, device)
        sums = data_batch.sum(1).unsqueeze(1)
        normalized_data_batch = data_batch / sums
        new_docs_prediction = etm_model.predict_proba(bow_tokens=normalized_data_batch)

    elif config_dict['optimization_params']['mode'] == 'eval':
        with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(device)
        model.eval()

        print('Visualizing model quality before training...')
        etm_model.eval()
        etm_model.print_words_per_topic(words_amount=config_dict['evaluation_params']['num_words'],
                                        vocab=vocab, lang='en')
        with torch.no_grad():
            # get document completion perplexities
            test_ppl = model.evaluate(source='test', test_1_tokens=test_1_tokens, test_1_counts=test_1_counts,
                                      test_2_tokens=test_2_tokens, test_2_counts=test_2_counts,
                                      train_tokens=train_tokens, vocab=vocab, tc=config_dict['evaluation_params']['tc'],
                                      td=config_dict['evaluation_params']['td'])

            # get most used topics
            indices = torch.tensor(range(config_dict['num_docs_train']))
            indices = torch.split(indices, config_dict['batch_size'])
            thetaAvg = torch.zeros(1, config_dict['model_params']['num_topics']).to(device)
            thetaWeightedAvg = torch.zeros(1, config_dict['model_params']['num_topics']).to(device)
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(train_tokens, train_counts, ind, config_dict['vocab_size'], device)
                sums = data_batch.sum(1).unsqueeze(1)
                cnt += sums.sum(0).squeeze().cpu().numpy()
                if config_dict['optimization_params']['bow_norm']:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                theta, _ = model.get_theta(normalized_data_batch)
                thetaAvg += theta.sum(0).unsqueeze(0) / config_dict['num_docs_train']
                weighed_theta = sums * theta
                thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
                if idx % 100 == 0 and idx > 0:
                    print('batch: {}/{}'.format(idx, len(indices)))
            thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
            print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

            # show topics
            etm_model.print_words_per_topic(words_amount=config_dict['evaluation_params']['num_words'], vocab=vocab)
            etm_model.print_words_per_topic(words_amount=config_dict['evaluation_params']['num_words'],
                                            vocab=vocab, lang='en')


    """
    # in order to see the embeddings of the topics: m.alphas.weight
    # in order to see the embeddings of the words: m.rho.weight
    #saving the output of a model as files (for visualization purposes)
    import numpy as np
    import csv
    import pandas as pd
    
    rho_as_numpy = model.rho.weight.data.cpu().detach().numpy()
    alphas_as_numpy = model.alphas.weight.data.cpu().detach().numpy()
    all_embeddings = np.concatenate((rho_as_numpy, alphas_as_numpy))
    np.savetxt("all_embeddings.tsv", all_embeddings, delimiter="\t")
    
    topics_list = ['topic_' + str(i) for i in range(model.num_topics)]
    
    vocab_with_topics = vocab
    vocab_with_topics.extend(topics_list)
    with open('vocab.tsv', 'w', newline='\n', encoding="utf-8") as f_output:
        for cur_word in vocab:
            f_output.write(cur_word + '\n')
    
    # same thing for English
    translator = Translator()
    vocab_translated = list()
    for cur_idx, cur_word in enumerate(vocab):  # topic_indices:
        if cur_idx%100 == 0:
            print(f"Passed over {cur_idx} items. Let's continue")
        result = translator.translate(cur_word, src='ar')
        vocab_translated.append(result.text)
    vocab_translated.extend(topics_list)
    vocab_both_lang = pd.DataFrame({'arabic': vocab_with_topics, 'english': vocab_translated})
    vocab_both_lang.to_csv("vocab_both_lang.tsv", sep='\t', index=False)
    """