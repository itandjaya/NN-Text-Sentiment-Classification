# NN-Text-Sentiment-Classification
Analyzing IMDB movie reviews (text) to predict whether it's a positive or negative review.


Dataset pre-processing: Use a pre-trained text embedding model from TensorFlow Hub:

google/tf2-preview/gnews-swivel-20dim-with-oov/1 - same as google/tf2-preview/gnews-swivel-20dim/1, but with 2.5% vocabulary converted to OOV buckets. This can help if vocabulary of the task and vocabulary of the model don't fully overlap.
google/tf2-preview/nnlm-en-dim50/1 - A much larger model with ~1M vocabulary size and 50 dimensions.
google/tf2-preview/nnlm-en-dim128/1 - Even larger model with ~1M vocabulary size and 128 dimensions.
