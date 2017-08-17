import pandas as pd
import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from gensim.models import word2vec
from sklearn.model_selection import KFold
from collections import Counter
from gensim import models
import gensim
from keras.utils import np_utils
from sklearn.metrics import f1_score
import os
from keras.models import model_from_json
from glove import Corpus, Glove
from sklearn.model_selection import train_test_split

vocab_path='/data/dchaudhu/metaphor2vec/data/vocab/corpus_vocab.npy'
word_vec_file='/data/dchaudhu/metaphor2vec/data/word_vecs/mit_metaphor/domain_vecs'
glove_word_vecs = '/data/dchaudhu/metaphor2vec/data/word_vecs/mit_metaphor/domain_vecs'
#pos_revw_file = '/data/dchaudhu/metaphor2vec/data/reviews/positive/positive_revws.tsv'
#neg_revw_file = '/data/dchaudhu/metaphor2vec/data/reviews/negative/negative_revws.tsv'
common_w_f = '/data/dchaudhu/metaphor2vec/data/vocab/common_imp_words.txt'
lda_model = '/data/dchaudhu/metaphor2vec/data/lda_models/amazon_lda'
saved_dict = '/data/dchaudhu/metaphor2vec/data/lda_models/amazon_dict'
reviews = '/data/dchaudhu/metaphor2vec/data/reviews/'
lda_nn = '/data/dchaudhu/metaphor2vec/data/lda_models/lda_nn/'

domains = ['electronics', 'books', 'kitchen_n_housewares', 'dvd']
common_words = [l.strip() for l in open(common_w_f)]

#token level separation
def replace_word(sentence, lda, dictionary, model):
    replaced_sent = []
    sent_lda_vec = get_lda_vec(lda[dictionary.doc2bow(sentence)]).reshape((1,100))
    sent_domain = domains[np.argmax(model.predict(sent_lda_vec))]
    for word in sentence:
        if word in common_words:
            word = word.replace(word, word + '_' + str(sent_domain))
            replaced_sent.append(word)
        else:
            replaced_sent.append(word)

    return replaced_sent

#get vectors
def get_index_to_embeddings_mapping(vocab, word_vecs):
    embeddings = {}
    for word in vocab.keys():
        try:
            #embeddings[word] = word_vecs.word_vectors[word_vecs.dictionary[word]]A
	    embeddings[word] = word_vecs[word]
        except KeyError:
            # map index to small random vector
            # print "No embedding for word '"  + word + "'"
            embeddings[word] = np.random.uniform(-0.01, 0.01, 300)
    return embeddings

#get alpha vector from lda model for a document
def get_lda_vec(lda_dict):

    lda_vec = np.zeros(100)
    for id, val in lda_dict:
        lda_vec[id] = val
    return lda_vec


def get_word2id(word, w2idx_dict):
    try:
        return w2idx_dict[word]
    except KeyError:
        return len(w2idx_dict)

#getting domain reviews from file
def get_domain_revs(domain, lda, dictionari, word2idx, model):
	
    
    print "Getting reviews for domain:" + domain

    pos_revw_file = reviews + domain + '/' + 'positive.tsv'
    neg_revw_file = reviews + domain + '/' + 'negative.tsv'
    positive_reviews = pd.read_table(pos_revw_file, sep='\t')
    negative_reviews = pd.read_table(neg_revw_file, sep='\t')

    positive_reviews['label'] = 1
    negative_reviews['label'] = 0

    #positive_reviews = np.array(positive_reviews)
    #negative_reviews = np.array(negative_reviews)

    review_text = np.concatenate((np.array(positive_reviews['review']), np.array(negative_reviews['review'])), axis=0)

    review_labels = np_utils.to_categorical(np.concatenate((np.array(positive_reviews['label']), np.array(negative_reviews['label'])), axis=0))
  
    review_text = [sent.split() for sent in review_text]
    print "Sample review text:"
    print review_text[0][0:5]

    if os.path.exists('/data/dchaudhu/metaphor2vec/data/vocab/' + domain + '/reviews_replaced.npy'):
	review_text = np.load('/data/dchaudhu/metaphor2vec/data/vocab/' + domain + '/reviews_replaced.npy')
    else:
        review_text = [replace_word(sent, lda, dictionari, model) for sent in review_text]
    #review_text = np.load('/data/dchaudhu/metaphor2vec/data/vocab/reviews_replaced.npy')
        np.save('/data/dchaudhu/metaphor2vec/data/vocab/' + domain + '/reviews_replaced', review_text)
    print "number of reviews: " + str(len(review_text))
    print "Sample review texts after replace...."
    print review_text[0][0:5]


    review_text = np.array([[get_word2id(word, word2idx) for word in sent] for sent in review_text])
    
    print review_text.shape, review_labels.shape 
    return review_text, review_labels


#the cnn model architecture
def cnn_model(embedding_weights, cv_dat, max_len):
    max_len = 1500 if max_len > 1500 else max_len
    dropout = 0.5

    train_x, test_x, train_y, test_y = cv_dat

    print "Maximum length of sentence:" + str(max_len)
    print "Distribution of labels in training set:"
    print Counter([np.argmax(lbl) for lbl in train_y])
    print "Distribution of labels in testing set:"
    print Counter([np.argmax(lbl) for lbl in test_y])


    print train_x.shape
    train_x = np.array(sequence.pad_sequences(train_x, maxlen=max_len, padding = 'post'), dtype=np.int)
    test_x = np.array(sequence.pad_sequences(test_x, maxlen=max_len, padding = 'post'), dtype=np.int)

    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.1, stratify=train_y)
    print (train_x.shape)
    print train_y.shape
    print Counter([np.argmax(lbl) for lbl in val_y])
    """
    lda_input = Input(shape=(100,), dtype ='float32', name = "lda_input")
    lda_dense = Dense(500, activation='tanh', kernel_initializer="glorot_uniform")(lda_input)
    lda_drop = Dropout(dropout)(lda_dense)
    lda_out = Dense(4, 'softmax')(lda_drop)
    """
    review_text = Input(shape=(max_len, ), dtype='int64', name="body_input")
    embedded_layer_body = Embedding(embedding_weights.shape[0], embedding_weights.shape[1], mask_zero=False,
                                    input_length=max_len, weights=[embedding_weights], trainable=True)(review_text)
    conv1 = Conv1D(filters=256, kernel_size=1, padding='same', activation='relu', kernel_initializer="glorot_uniform")
    conv2 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer="glorot_uniform")
    conv3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_initializer="glorot_uniform")

    conv1a = conv1(embedded_layer_body)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv2a = conv2(embedded_layer_body)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv3a = conv3(embedded_layer_body)
    glob3a = GlobalAveragePooling1D()(conv3a)

    merge_pooling = concatenate([glob1a, glob2a, glob3a])
    print merge_pooling

    hidden_layer = Dense(1200, activation='tanh', kernel_initializer="glorot_uniform")(merge_pooling)
    dropout_hidden = Dropout(dropout)(hidden_layer)

    hidden_layer_2 = Dense(600, activation='tanh', kernel_initializer="glorot_uniform")(dropout_hidden)
    dropout_hidden_2 = Dropout(dropout/2)(hidden_layer_2)

    output_layer = Dense(2, activation='softmax')(dropout_hidden_2)

    model = Model([review_text], output=output_layer)

    adam = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(loss=ncce, optimizer=adam, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3,
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]
    print model.summary()
    model.fit([tr_x], tr_y, batch_size=64, epochs=30,
              verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=[val_x, val_y])
    test_predictions = model.predict([test_x], verbose=False)
    print test_predictions[0]
    #test_y = [np.argmax(pred) for pred in test_y]
    test_pred = [np.argmax(pred) for pred in test_predictions]
    test_y = [np.argmax(label) for label in test_y]
    acc = accuracy_score(test_y, test_pred)
    print "Accuracy on test fold:"
    print acc
    return acc


#run cross-domain sentence classification
def run_cnn_model():

    #Loading the vocabulary for all domains
    print "loading the vocabulary..."
    vocab = np.load(vocab_path).item()
    #word to index mapping, +1 for oov words
    word2idx = dict(zip(vocab.keys(), range(0, len(vocab) + 1)))

    print len(word2idx)

    print "Loading the LDA model....."
    lda = models.LdaModel.load(lda_model)
    dictionri = gensim.corpora.Dictionary.load(saved_dict)

    print "Loading the word_vecs"
    corpus_wordvec = word2vec.Word2Vec.load(word_vec_file)
    #corpus_wordvec =  Glove.load(glove_word_vecs)
    index_to_vector_map = get_index_to_embeddings_mapping(word2idx, corpus_wordvec)

    n_symbols = len(word2idx) + 1  # adding 1 to account for masking
    embedding_weights = np.zeros((n_symbols, 300))

    for word, index in word2idx.items():
        try:
            embedding_weights[index, :] = index_to_vector_map[word]
        except KeyError:
            embedding_weights[index, :] = np.random.uniform(-0.01, 0.01, 300)

    """
    positive_reviews = pd.read_table(pos_revw_file, sep='\t')
    negative_reviews = pd.read_table(neg_revw_file, sep='\t')

    positive_reviews['label'] = 1
    negative_reviews['label'] = 0

    #positive_reviews = np.array(positive_reviews)
    #negative_reviews = np.array(negative_reviews)

    review_text = np.concatenate((np.array(positive_reviews['review']), np.array(negative_reviews['review'])), axis=0)

    print "Sample reviews"
    print review_text[0]

    #load the lda model and the corpus dictionary..
    print "Loading the LDA model....."
    lda = models.LdaModel.load(lda_model)
    dictionri = gensim.corpora.Dictionary.load(saved_dict)

    #split review text into words
    review_text = [sent.split() for sent in review_text]
    print "Sample review text:"
    print review_text[0]

    #replace words in review texts
    print "Replacing metaphors with domain knowledge...."
    #review_text = [replace_word(sent, lda, dictionri) for sent in review_text]
    review_text = np.load('/data/dchaudhu/metaphor2vec/data/vocab/reviews_replaced.npy')
    #np.save('/data/dchaudhu/metaphor2vec/data/vocab/reviews_replaced', review_text)

    print "Sample review texts after replace...."
    print review_text[0]

    #get review labels
    review_labels = np_utils.to_categorical(np.concatenate((np.array(positive_reviews['label']), np.array(negative_reviews['label'])), axis=0))

    print "review labels:"
    print review_labels

    review_text = np.array([[get_word2id(word, word2idx) for word in sent] for sent in review_text])
    """
    
    print "Loading the neural network model.."
    #load the neural net model
    json_file = open(lda_nn+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(lda_nn+"model.h5")
    print("Loaded model from disk")
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    test_acc = 0.0
 
    for domain in domains:
        x_tr = []
    	y_tr = []
        train_domains = [dom for dom in domains if dom != domain]
        print train_domains
        for dom in train_domains:
           revws, labels = get_domain_revs(dom, lda, dictionri, word2idx, model)
           #print labels
           x_tr.append(revws)
           y_tr.append(labels)
        y_tr = np.array(y_tr).reshape(6000,2)
        
        x_te, y_te = get_domain_revs(domain, lda, dictionri, word2idx, model)
        print "length of training samples:" + str(len(x_tr))
        if domain == 'books':
		max_sent_len = 1500
	else:
           	max_sent_len = np.max([len(sent) for sent in x_te]) 
        cv_dat = np.array(x_tr).flatten(), x_te, (np.array(y_tr)), y_te
        print "Leave out domain: " + domain
    	test_acc = test_acc + cnn_model(embedding_weights, cv_dat, max_sent_len)
    
    print "Done training the model, average accuracy on CV set:"
    print test_acc/4

    #max_sent_len = np.max([len(sent) for sent in review_text])

    #doing kfold cross validation
    """ 

    kf = KFold(n_splits=10, shuffle=True, random_state=666)

    test_acc = 0.0

    for train, test in kf.split(review_labels):
        x_tr, x_te, y_tr, y_te = review_text[train], review_text[test], review_labels[train], review_labels[test]
        cv_dat = x_tr, x_te, y_tr, y_te

        print "Training the model now......"
        test_acc = test_acc + cnn_model(embedding_weights, cv_dat, max_sent_len)

    print "Done training the model, average accuracy on CV set:"
    print test_acc/10
    """

if __name__ == '__main__':
    run_cnn_model()

