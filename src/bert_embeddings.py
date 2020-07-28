import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import tokenization
from bert import run_classifier
from bert import optimization
# from kneed import KneeLocator
# from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.metrics.pairwise import cosine_similarity
# from utils.stats_utils import *
from nlp_utils import *

class BERT(object):
    def __init__(self):
        
        self.ABBREVIATION = ["Sr.", "i.e."]
        self.SKIP = []
        self.BERT_URL = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
        self.bert_module = hub.Module(self.BERT_URL)   
    
    # return a tuple (text, indicator, gap_score). 
    # If indicator is false, no best k was found. 
    # If gap_score was not positive, summarization is not valid. 
        
         
    def bert_embeddings(self, sentences):
        # input: a list of string
        module = self.bert_module
        tokenizer = self.create_tokenizer_from_hub_module()

        input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences = sentences, \
                                                            max_seq_len = 128, \
                                                            tokenizer = tokenizer)
        input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
        input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
        segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)

        bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, \
                                                input_mask: input_mask_vals, segment_ids: segment_ids_vals})

        return out['pooled_output']


    def create_tokenizer_from_hub_module(self):
        tokenization_info = self.bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    def batch(self,iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


def similar_words_extraction(i,ideal_sentences,indices,five_words_list):
  similar_words=[five_words_list[x] for x in indices[i]]
  df=pd.DataFrame(similar_words,columns=['Similar_words'])
  df['Ideal_sentence']=ideal_sentences[i]
  return df
    
# df=pd.read_csv('/content/drive/My Drive/Identifying_objectionable_phrases/Five_word_phrases.csv')
# # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# # df.rename(columns={'Three word phrases':'Three_word_phrases'},inplace=True)
# sentences=list(df.iloc[:,0])
# print(len(sentences))
# sentences=sentences[:8000]
# model=BERT()
# # Extracting the features for three word phrases
# features=[]
# for i in model.batch(sentences,500):
#   features.append(model.bert_embeddings(i))

# # Creating a flat list from the features list
# flat_features_list = []
# for sublist in features:
#     for item in sublist:
#         flat_features_list.append(item)

# # Creating a data frame for the flattened features
# features_df=pd.DataFrame(flat_features_list)

# # Exporting data frame to .csv file
# features_df.to_csv('/content/drive/My Drive/Features_extracted/Extracted_features_for_five_words.csv',index=False,header=True)

# Calculating features for ideal sentences
ideal_sentences=['No significant changes outside of expected assay variability were observed',
'No adverse trend was observed in main peak or acidic region by IE-HPLC','No adverse trend was observed in HMWS by SE-HPLC.',
'A slight increase in HMWS of approximately 0.2%-0.3% may be observed by SE-HPLC after storage at -20°C for up to 60 months, with a corresponding decrease in the main peak.',
' This change is just outside assay variability.']

model=BERT()
ideal_features=model.bert_embeddings(ideal_sentences)
ideal_features_df=pd.DataFrame(ideal_features)

#  Exporting to .csv file
ideal_features_df.to_csv('/Users/soodk2/Documents/Extracted_files/Feature_vector_ideal_sentences.csv',index=False,header=True)



#################################---------Finding similar words for 5 WORD PHRASES------------################
# Importing feature vectors
# five_word_features_matrix=pd.read_csv('/content/drive/My Drive/Features_extracted/Extracted_features_for_five_words.csv').to_numpy()
# ideal_features_matrix=pd.read_csv('/content/drive/My Drive/Features_extracted/Ideal_features.csv').to_numpy()


# # Calculating cosine similarity
# similarity_matrix=cosine_similarity(ideal_features_matrix,five_word_features_matrix)

# # Exporting similarity scores to dataframe
# similarity_df=pd.DataFrame(similarity_matrix)

# # Exporting similarity data frame to .csv file
# similarity_df.to_csv('/content/drive/My Drive/Similarity_scores/Five_word_similarity.csv',index=False,header=True)

# # Calculating indices of similarity score greater than threshold of 0.8
# score_list=[]

# # Extracting the similar words
# for i in similarity_matrix:
#     score_dict={}
#     for j in range(len(i)):
#         if i[j]>0.8:
#             score_dict[j]=i[j]
#     score_list.append(score_dict)

# # Reading the five word phrases into data frame
# df=pd.read_csv('/content/drive/My Drive/Identifying_objectionable_phrases/Five_word_phrases.csv')

# # Reshuffling the dataframe
# df = df.sample(frac=1).reset_index(drop=True)

# five_words_list=list(df.iloc[:,0])
# print(len(five_words_list))

# # Filtering on basis of index for the first ideal sentence
# df_list=[]
# for i in range(7):
#   df_list.append(similar_words_extraction(i,ideal_sentences,indices,five_words_list))

# # Concatenating all the dataframes
# df_concatenated=pd.concat(df_list,ignore_index=True)

# # Shuffling the data frame
# df_concatenated = df_concatenated.sample(frac=1).reset_index(drop=True)

# # Printing the shape and head of the main data frame
# print(df_concatenated.shape)
# print(df_concatenated.head())

# # Exporting the results into dataframe
# df_concatenated.to_csv('/content/drive/My Drive/Similarity_dataframe_results/Five_length_similar_words.csv',index=False,
#                        header=True)



