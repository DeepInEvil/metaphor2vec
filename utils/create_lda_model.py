import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#get stopwords
stop = stopwords.words('english')
#read all data
data = pd.read_table('../unlabelled_merged.tsv', sep='\t')
data = np.array(data)
data = [sent[0].split() for sent in data]
data = [[word for word in sent if word not in stop] for sent in data]
print data[0]
data = [[word for word in sent if word not in ('\\)', '\\(')] for sent in data]

#create the dictionary
dictionary = corpora.Dictionary(data)
dictionary.save('amazon_dict')

#create the bow model
bow_dat = [dictionary.doc2bow(sent) for sent in data]

#create the lda model
lda = models.LdaMulticore(bow_dat, id2word=dictionary, num_topics=4, workers=30, passes=20)
lda.save('amazon_lda')

