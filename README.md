# MovieSumPrediction
 using cmu plot-genre dataset:
 * Predict applicable genres
 * Find similar movies

# Notebooks
Notebooks contains 2 files:
* Deep Learning (transfer learning on BERT)
* Shallow Learning Exploration

# Execution
* python predict.py [input_file_name]

# Process
I explored the data with simple graphs, using matplotlib and nltk. Using a visual inspection informed by genre count, I merged a few (semantically) duplicate ones. Note that the approach to finding similar movies is using a cosine similarity on the embedding of the plot text into the space determined by the requisite model.

## Deep Learning
Multilabel Transformer finetuning

### Modeling
This was done using a huggingface implimentation of tokenizer and BERT. Using a BERT Transformer, I created a multi-label problem by using a BinaryCrossentropy Loss on model output. The prediction is done using the sigmoid of the outputs, and thresholding the scores of the output vector to determine which genres matched. Alternatively you can rank and use the top k.

### TODOs
Exploring finetuning the BERT Model on a Masked Language Model (or similar) objective would have given a better outcome, where the embedding vectors that we use for prediction, would be more attuned to the specific meaning in the context of movies. I believe that this is the reason the shallow learning did just as well or better.

## Shallow Learning
LogisticRegression (with various regularization), Naive Bayes Classification

### Analysis Process
Here we actually cleaned out the stopwords ourselves, and used a TfIDf vectorization approach to embedding the plot text. For each of the shallow learners, a One Vs Rest Classifier is Trained. This means that for the 344 Genres, there was 1 model for each of them classifying genre vs not genre.

### TODOs
Exploring the relationship between word apperance and genre would have been nice. Additionally, dropping the genres which only have a few samples would have drastically decreased the number of labels. In addition, visualizing the ROC and AU(ro)C would have been informative on how well the training data generalized to the testing data.
