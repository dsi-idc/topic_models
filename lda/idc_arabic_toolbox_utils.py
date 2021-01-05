class arab_toolbox:
    
    def __init__(self):
        pass

    def load_pickle(file_path):
        
        import pickle

        text_all_dict = pickle.load( open( file_path, "rb" ) )
        
        print("number of tweets is: " + str(len(text_all_dict)))
        
        return text_all_dict
    
    
    def retweet_counts(text_dicts):
        
        main_text_keys = list(text_dicts.keys())

        list_of_count_dicts = []

        for i in range(0,len(main_text_keys)):

            responses = text_dicts[main_text_keys[i]]["responses"]

            responses_keys = list(responses.keys())

            list_of_count_dicts.append({"retweet_count":len(responses_keys)})

        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        count_df = pd.DataFrame(list_of_count_dicts)

        sns.distplot(count_df["retweet_count"])
    
    
    def clean_text_list(text_dicts, thread_level=False, remove_stop=True):
        
        # keys are threads
        
        main_text_keys = list(text_dicts.keys())
        
        # create a list of texts from the json data that contains not just texts
        
        text_list = []
        
        
        
        # thread or not thread level to create the text list
        
        
        if thread_level==True:
            
            for i in range(0,len(main_text_keys)):

                main_tweet = text_dicts[main_text_keys[i]]["main_post"]["text"]

                responses = text_dicts[main_text_keys[i]]["responses"]

                responses_keys = list(responses.keys())

                for response_key in responses_keys:

                    one_response = responses[response_key]["text"]

                    main_tweet = main_tweet+" "+one_response

                    main_tweet = main_tweet.strip()
                    
                text_list.append(main_tweet)

        else:

            for i in range(0,len(main_text_keys)):

                text_list.append(text_dicts[main_text_keys[i]]["main_post"]["text"])
        
        
        # tokenization + stopwords, alphanumeric cleaning

        import nltk

        #nltk.download('punkt')

        from nltk import word_tokenize,sent_tokenize

        # I am removing signs at this point and short words. If stop_words is true then also removing those
        
        if remove_stop==True:
        
            import pickle
            import pandas as pd
            stop_words = pd.read_excel("arab_stop_words.xlsx")["stop_words"].tolist()
            
            def clean(text):
                text_clean_list = [w.strip() for w in word_tokenize(text) if w.isalpha() and w not in stop_words and len(w)>1]
                return text_clean_list
        
        else:
            
            def clean(text):
                text_clean_list = [w.strip() for w in word_tokenize(text) if w.isalpha() and len(w)>1]
                return text_clean_list
        
        text_list = [clean(text) for text in text_list]
        
        return text_list
    
    
    
    def pos_clean(text_list,java_path,stanford_tagger_path,remove_verbs=True):
        
        import os
        #java_path = "C:/Program Files/Java/jdk1.8.0_261/bin/java.exe"
        os.environ['JAVAHOME'] = java_path
        #stan_path = "C:/Users/אילנה/Dropbox/jupyter_notebooks/data-science/idc-research/mine/stanford-tagger-4.0.0/"
        
        from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
        
        arabic_postagger = POS_Tag(stanford_tagger_path+'models/arabic.tagger', stanford_tagger_path+'/stanford-postagger.jar')
        
        text_list_pos = [arabic_postagger.tag(inner_word_list) for inner_word_list in text_list]
        
        
        if remove_verbs==True:
            
            pos_to_remove = ["VB","VBD","VBG","VBN","VBP","VBZ"]

            text_list_final = []

            for inner_list in text_list_pos:

                final_inner_list = []

                for pos_tuple in inner_list:

                    # sometimes structure is unstable, so need to use find

                    if pos_tuple[0].find("/")>=0:

                        idx=0

                    else:

                        idx=1

                    if pos_tuple[idx].split("/")[1] not in pos_to_remove:           

                        final_inner_list.append(pos_tuple[idx].split("/")[0])

                text_list_final.append(final_inner_list)
                
            else:
                
                text_list_final = text_list
                
                
        return text_list_final
        
    
        
    def create_ngram(text_list_final,threshold="low"):

        # bigram/trigram

        import gensim

        # Build the bigram and trigram models
        # trigram checks for one more word which co-occurs with a bigram.
        bigram = gensim.models.Phrases(text_list_final, min_count=5, threshold=10) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[text_list_final], threshold=10)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)


        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]


        text_list_final = make_bigrams(text_list_final)

        text_list_final = make_trigrams(text_list_final)


        return text_list_final
        
        
    
    def text_list_translator(text_list,language='en'):
        
        import time
        
        # test translator

        from googletrans import Translator

        translator = Translator()
        

        text_list_trans = []

        for i in range(0,len(text_list)):
            
            time.sleep(5)

            inner_translated_list = [translator.translate(w,dest=language).text for w in text_list[i]]

            text_list_trans.append(inner_translated_list)
            
        return text_list_trans
            
    
    def create_bow_corpus(text_list,print=True):
        
        from gensim.corpora.dictionary import Dictionary

        dictionary = Dictionary(text_list)
        
        corpus = [dictionary.doc2bow(doc) for doc in text_list]
        
        return dictionary,corpus
        
    
    
    def buildLDA(dictionary,corpus,num_topics):
        import gensim
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word = dictionary,
                                                   num_topics=num_topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)
        return lda_model
    
    
    
def lda_vis(lda_model, corpus, dictionary):

    # visualize the topics and words

    import pyLDAvis
    import pyLDAvis.gensim  # don't skip this
    import matplotlib.pyplot as plt
    #%matplotlib inline

    pyLDAvis.enable_notebook()

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

    vis

    return vis


def print_bow(dictionary,corpus):
    
    import time
    
    for i in range(0,len(corpus)):
        
        time.sleep(0.1)

        doc = corpus[i]

        doc = sorted(doc, key=lambda w: w[1], reverse=True)


        for word_id, word_count in doc[:10]:

            print(dictionary.get(word_id), word_count)

        print("----------------------------------------")

        
def print_tfidf(dictionary,corpus):

    from gensim.models.tfidfmodel import TfidfModel

    tfidf = TfidfModel(corpus)
    
    import time
    
    for i in range(0,len(corpus)):
        
        time.sleep(0.1)

        tfidf_weights_doc = tfidf[corpus[i]]

        tfidf_weights_doc_sorted = sorted(tfidf_weights_doc, key=lambda w: w[1], reverse=True)


        for term_id, weight in tfidf_weights_doc_sorted[:10]:

            print(dictionary.get(term_id), weight)

        print("---------------------------------------")

