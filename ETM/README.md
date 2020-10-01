# Arabic ETM
This code is an arabic version of the original ETM paper. We have changed it a bit to work with arabic datasets for IDC purposes

## running the code
We have created a file called 'stops_arabic.txt' which is used for providing stop-words in arabic, you should have it
under the 'scipts' folder
+ You should first run the 'data_intuview_arabic.py' code, to generate data for the algorithm
+ Next, you should run the 'main.py' code, with the settings for training (--mode train, --dataset intuview --data_path data/intuview and so on)
+ Lastly, you can run the 'main.py' code, with the settings for evaluation (e.g., mode==eval)
Note that in eval mode, you need to define the location of the trained model. Here is an example of the full configuration parameters:
--mode val --dataset intuview --data_path data/intuview --num_topics 20 --train_embeddings 1 --epochs 10 --load_from etm_intuview_K_20_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_1

# ETM

This is code that accompanies the paper titled "Topic Modeling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. (Arxiv link: https://arxiv.org/abs/1907.04907)

ETM defines words and topics in the same embedding space. The likelihood of a word under ETM is a Categorical whose natural parameter is given by the dot product between the word embedding and its assigned topic's embedding. ETM is a document model that learns interpretable topics and word embeddings and is robust to large vocabularies that include rare words and stop words.

## Dependencies

+ python 3.6.7
+ pytorch 1.1.0

## Datasets

All the datasets are pre-processed and can be found below:

+ https://bitbucket.org/franrruiz/data_nyt_largev_4/src/master/
+ https://bitbucket.org/franrruiz/data_nyt_largev_5/src/master/
+ https://bitbucket.org/franrruiz/data_nyt_largev_6/src/master/
+ https://bitbucket.org/franrruiz/data_nyt_largev_7/src/master/
+ https://bitbucket.org/franrruiz/data_stopwords_largev_2/src/master/ (this one contains stop words and was used to showcase robustness of ETM to stop words.)
+ https://bitbucket.org/franrruiz/data_20ng_largev/src/master/

All the scripts to pre-process a given dataset for ETM can be found in the folder 'scripts'. The script for 20NewsGroup is self-contained as it uses scikit-learn. If you want to run ETM on your own dataset, follow the script for New York Times (given as example) called data_nyt.py  

## To Run

To learn interpretable embeddings and topics using ETM on the 20NewsGroup dataset, run
```
python main.py --mode train --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --epochs 1000
```

To evaluate perplexity on document completion, topic coherence, topic diversity, and visualize the topics/embeddings run
```
python main.py --mode eval --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --tc 1 --td 1 --load_from CKPT_PATH
```

To learn interpretable topics using ETM with pre-fitted word embeddings (called Labelled-ETM in the paper) on the 20NewsGroup dataset:

+ first fit the word embeddings. For example to use simple skipgram you can run
```
python skipgram.py --data_file PATH_TO_DATA --emb_file PATH_TO_EMBEDDINGS --dim_rho 300 --iters 50 --window_size 4 
```

+ then run the following 
```
python main.py --mode train --dataset 20ng --data_path data/20ng --emb_path PATH_TO_EMBEDDINGS --num_topics 50 --train_embeddings 0 --epochs 1000
```

## Citation

```
@article{dieng2019topic,
  title={Topic modeling in embedding spaces},
  author={Dieng, Adji B and Ruiz, Francisco J R and Blei, David M},
  journal={arXiv preprint arXiv:1907.04907},
  year={2019}
}
```

