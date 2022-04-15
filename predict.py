
import transformers
import sys
import os
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from scipy import spatial
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

# Constants
THRESHOLD_CLF_BERT = 0.25
THRESHOLD_MIN_BERT = 0.1

class TFBertForMultilabelClassification(TFBertPreTrainedModel):
    model_name='bert-base-uncased'
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

if __name__=='__main__':
    try:
        plot_example = open(sys.argv[1], 'r').read()
    except:
        raise Exception ("Please add the location of a txt file containing a plot synopsis:\n bash$: python3 predict.py [input_file.txt]")

    models_dir = './models/'
    print ("Down/Loading Models...")
    tokenizer = transformers.AutoTokenizer('bert-base-uncased')
    multi_label_model = TFBertForMultilabelClassification.from_pretrained('dataartist/movie_genre_bert')
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    train = pd.read_csv('./data/movie_short.csv')

    print("Calculating Embeddings...")
    embedding = embedding_model.encode(train['plot'].values,normalize_embeddings=True, show_progress_bar=True)

    print ("Generating Genre Predictions using fine-tuned BERT Transformer...")
    plot_example_tok = tokenizer(plot_example, return_tensors="tf")
    genre_classification_logits = multi_label_model(plot_example_tok)
    genre_results = tf.nn.sigmoid(genre_classification_logits).numpy()[0]
    genre_results[np.where(np.round(genre_results,1) < THRESHOLD_MIN_BERT)[0]] = 0
    preds = np.where(genre_results > THRESHOLD_CLF_BERT)[0]
    if len(preds) < 1:
        preds = [np.argmax(genre_results)]
    preds = sorted(zip(genre_results[preds] , mlb.classes_[preds]), key=lambda p: p[0])
    if round(preds[0][0],1) < THRESHOLD_MIN_BERT:
        print ("No Good Genre Predictions.")
    else:
        print ("Genre(s):")
        for p in preds:
            print ('Genre: {} Score: {}'.format(p[1],p[0]))

    tree = spatial.KDTree(np.vstack(embedding))
    vec = embedding_model.encode(plot_example,normalize_embeddings=True)
    ed, pos = tree.query(vec,5)
    #tree.query()
    ed/=2 #ed = 2 - 2*cos, we want 1-cos for a normalized score
    print ("5 most Similar Movies:")
    for p in pos:
        print (train.movie_name.tolist()[p])
