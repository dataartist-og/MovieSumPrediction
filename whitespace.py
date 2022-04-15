# -*- coding: utf-8 -*-
"""Whitespace.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TSKy8_eZE3b93cP9Df_g9uhLQUHr6CIm
"""





# Commented out IPython magic to ensure Python compatibility.
# %%shell
# #wget https://bit.ly/3lH1hKU 
# pip install bert-tensorflow sentencepiece
#

#pip install -U sentence-transformers
#pip install langdetect

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2



# Commented out IPython magic to ensure Python compatibility.
# %%shell
# #cp /content/drive/MyDrive/kaggle.json /root/.kaggle/
# #kaggle datasets download -d rounakbanik/the-movies-dataset
# #unzip the-movies-dataset.zip

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# #unzip /content/MovieSummaries.tar.gz
# #7z x  /content/MovieSummaries.tar.gz
# #tar -xvzf  /content/MovieSummaries.tar.gz
# pip3 install transformers

import transformers

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nltk
import joblib

# %matplotlib inline

#data = pd.read_csv("movies_metadata.csv", sep = ',', header = 0, low_memory=False)
#movies = data[['title','overview','genres']].copy()
#movies.head()

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
pd.set_option('display.max_colwidth', 300)

file_path = '/content/drive/MyDrive/MovieSummaries/'
export_dir = './models/'

meta = pd.read_csv(file_path+"movie.metadata.tsv", sep = '\t',header=None)
meta.head()

# rename columns
meta.columns = ["movie_id",1,"movie_name",3,4,5,'language','country',"genre"]

from ast import literal_eval

meta.language = meta.language.apply(lambda s: list(literal_eval(s).values()))





meta.head()

import csv
from tqdm.notebook import tqdm

plots = []

with open(file_path+"plot_summaries.txt", 'r') as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            if len(row) < 2: continue
            plots.append(row)





plots_df = pd.DataFrame(plots)

plots_df.columns = ['movie_id','plot']

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(plots_df, meta[['movie_id', 'movie_name','language', 'genre']], on = 'movie_id')

movies.head()



gs = movies.genre.apply(json.loads)
gs = gs.apply(lambda s: s.values())



movies.genre = gs

# remove samples with 0 genre tags
movies_new = movies[movies.genre.apply(lambda s: len(s)!=0)].copy()
#movies_new.genre = movies_new.genre.apply(lambda s: list(s.values()))

movies_new.shape, movies.shape

movies_new.genre.values[0]



clusts = ( ('Animal Picture',  
 'Animals'),
 ('Anti-war', 
 'Anti-war film'), 
 ('Biographical film', 
 'Biography', 
 'Biopic [feature]'), 
 ('Comedy', 
 'Comedy film'),
 ('Coming of age', 
 'Coming-of-age film'), 
 ('Education', 
 'Educational'), 
 ('Filipino', 
 'Filipino Movies'), 
 ('Gay', 
 'Gay Interest', 
 'Gay Themed'), 
 ('Gross out', 
 'Gross-out film'), 
 ('Monster', 
 'Monster movie'), 
 ('Pornographic movie', 
 'Pornography'), 
 ('Prison', 
 'Prison film'), 
 ('Sci Fi Pictures original films', 
 'Science Fiction'), 
 ('Social issues', 
 'Social problem film'), 
 ('Superhero', 
 'Superhero movie'), 
 ('Sword and sorcery', 
 'Sword and sorcery films'), 
 ('Tamil cinema', 
 'Tollywood') )

for cl in tqdm(clusts):
  movies_new.genre = movies_new.genre.apply(lambda ss: [cl[0] if s in cl else s for s in ss])

# get all genre tags in a list
all_genres = []
movies_new.genre.apply(all_genres.extend)
len(set(all_genres))

all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()

#nltk.download('stopwords')



movies_new.genre.apply(len).plot(kind='hist',bins=20)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, token_type_ids, labels, is_real_example=True):
        self.input_ids = input_ids
        self.attention_mask = input_mask
        self.token_type_ids = token_type_ids
        self.labels = labels,
        self.is_real_example=is_real_example

def create_examples(df, labels_available=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid = row[0]
        text_a = row[1]
        if labels_available:
            labels = row[2]
        else:
            labels = np.zeros(len(label_list))
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples

train = movies_new[['movie_id','plot','genre']].dropna()

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

from pickle import dump,load


try:
  mlb = load(open(file_path+'label_binarizer.pkl', 'rb'))
except:
  mlb = mlb.fit(train.genre.values)
  dump(mlb, open(file_path+'label_binarizer.pkl', 'wb'))





train.genre = train.genre.apply(lambda s: mlb.transform([s]))

train.genre =train.genre.apply(lambda s: s.flatten())

import tensorflow as tf
from transformers import TFBertPreTrainedModel, TFBertMainLayer, TFBertModel
from transformers import BertTokenizer
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

model_name='bert-base-uncased'
label_list = list(set(all_genres))

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
#config.output_hidden_states = False
# Load BERT tokenizer
# Load the Transformers BERT model
config.problem_type='multi_label_classification'
config.num_labels=len(label_list)
#transformer_model = transformers.TFBertForSequenceClassification.from_pretrained(model_name, config = config)

#transformer_model.summary()

TRAIN_VAL_RATIO = 0.8
TEST_VAL_RATIO = 0.5

LEN = train.shape[0]
SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)
SIZE_TEST = int(LEN*(TRAIN_VAL_RATIO + (1-TRAIN_VAL_RATIO)*TEST_VAL_RATIO))

x_train = train[:SIZE_TRAIN]
x_test = train[SIZE_TRAIN:SIZE_TEST]
x_val = train[SIZE_TEST:]

train_examples = create_examples(x_train)
test_examples = create_examples(x_test)
val_examples = create_examples(x_val)

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Sentences are encoded by calling model.encode()
embedding = embedding_model.encode(train['plot'].values,normalize_embeddings=True, show_progress_bar=True)



train['embedding'] = [i for i in embedding]

from scipy import spatial
import scipy
tree = spatial.KDTree(np.vstack(embedding))
plot_example = 'the movie was about a dwarf who hides in a hole and fucks bitches all day long. He killed a nigger and then went to jail.'

vec = embedding_model.encode(plot_example,normalize_embeddings=True)

ed, pos = tree.query(vec,100)
#tree.query()
ed/=2 #ed = 2 - 2*cos, we want 1-cos for a normalized score

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 10
# We'll set sequences to be at most 256 tokens long.
MAX_SEQ_LENGTH = 256





def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length=512):
    """Converts examples to features using specified tokenizer

    Args:
        examples (list): Examples to convert.
        tokenizer (obj): The tokenzier object.
        label_list (list): A list of all the labels.
        max_sequence_length (int): Maximum length of a sequence

    Returns:
        tf.Dataset: A tensorflow dataset.
    """

    features = []
    for ex_index, example in tqdm(enumerate(examples)):
        #print(example.__dict__)
        # Encode inputs using tokenizer
        inputs = tokenizer.encode_plus(
            example.text_a,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        # Create features and add to feature list
        labels = example.labels
        assert (labels.shape==labels.squeeze().shape)
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=labels))
    # Generator for creating tensorflow dataset
    def gen():
        for ex in features:
            yield  ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                    np.array(ex.labels).flatten())

    return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([max_seq_length]),
              'attention_mask': tf.TensorShape([max_seq_length]),
              'token_type_ids': tf.TensorShape([max_seq_length])},
             tf.TensorShape([len(label_list)])))





class TFBertForMultilabelClassification(TFBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultilabelClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertModel.from_pretrained(model_name).layers[0]
        #self.bert.layers[0].trainable=False
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=tf.keras.initializers.truncated_normal(config.initializer_range),
                                                name='classifier')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)

import transformers

#emb = transformers.TFBertModel.from_pretrained(model_name, config=config)

model_name='bert-base-uncased'
model = TFBertForMultilabelClassification.from_pretrained(model_name,num_labels=len(label_list))
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Commented out IPython magic to ensure Python compatibility.
"""model = tf.keras.Sequential()
#model.add
model.add(tf.keras.layers.Dense(len(label_list),activation='linear',name='classifier'))


# Prepare training: instantiate optimizer, loss and learning rate schedule 
optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay([2500,5000,10000],[1e-3,5e-4,1e-4,5e-5]))
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric1 = tf.keras.metrics.CategoricalAccuracy()
metric2 = tf.keras.metrics.TopKCategoricalAccuracy()



# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[metric1,metric2])


# %load_ext tensorboard


import os, datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)



# %tensorboard --logdir logs
y = np.vstack(train['genre'].values)


# Train and evaluate model
history = model.fit(x=embedding, y=y, batch_size=4, callbacks=[tensorboard_callback], epochs=NUM_TRAIN_EPOCHS, validation_split=0.2, verbose=2)

# Save the trained model 
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
model.save_pretrained(export_dir)"""



# Get pretrained weights and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = TFBertForMultilabelClassification.from_pretrained(model_name, config=config)
#model = TFBertForMultilabelClassification
#model = ()
#model = transformers.TFBertForSequenceClassification.from_pretrained(model_name, config=config)

model.bert.trainable = False# = model.bert.layers[0]

model.summary()

# Convert examples to features 
train_dataset = convert_examples_to_features(train_examples, tokenizer, label_list, MAX_SEQ_LENGTH)
valid_dataset = convert_examples_to_features(val_examples, tokenizer, label_list, MAX_SEQ_LENGTH)
test_dataset = convert_examples_to_features(test_examples, tokenizer, label_list, MAX_SEQ_LENGTH)





# Shuffle train data and put into batches
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Prepare training: instantiate optimizer, loss and learning rate schedule 
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[2500,5000],
        values=[5e-4,2e-4,1e-4]
     )
    )
loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.CategoricalAccuracy(name='acc'),
      tf.keras.metrics.TopKCategoricalAccuracy(name='top-k')
]

from collections import Counter
ct = Counter(all_genres)
class_weights = {}
for i,c in enumerate(mlb.classes_):
  class_weights[i] = 1/ct[c]

# Compile the model
model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=METRICS)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import os, datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

# Train and evaluate model
history = model.fit(train_dataset,callbacks=[tensorboard_callback],class_weight=class_weights, epochs=5, validation_data=valid_dataset, verbose=2)

models_dir = os.path.join(file_path,export_dir)

# Save the trained model 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model_path = os.path.join(models_dir,model_name,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
model.save_pretrained(model_path)

import transformers

multi_label_model = TFBertForMultilabelClassification.from_pretrained(os.path.join(models_dir,'moviebert_new_labels'))

plot_example = """A secret agent is given a single word as his weapon and sent to prevent the onset of World War III. He must travel through time and bend the laws of nature in order to be successful in his mission."""
plot_example = train['plot'].loc[0]

plot_example_tok = tokenizer(plot_example, return_tensors="tf")
genre_classification_logits = multi_label_model(plot_example_tok)
genre_results = tf.nn.sigmoid(genre_classification_logits).numpy()[0]

movies_new.head(1)

mlb.inverse_transform(np.ceil(np.round(genre_results,1)))

train.columns

genre_classification_logits[0].shape

train['genre'].values[0].shape, len(mlb.classes_)



genre_results.flatten()[np.where(mlb.transform([[  'Drama',  'World cinema']])[0]==1)[0]]

genre_results.flatten().shape


