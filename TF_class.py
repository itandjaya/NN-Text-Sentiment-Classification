## TF_class.py
## Training model to predict movie text reviews: 1: good, 0: bad.

## Text classification using IMDB dataset:
## http://ai.stanford.edu/~amaas/data/sentiment/
## Source code: tfds.text.imdb.IMDBReviews


## Needed so the code works on older version of Python.
from __future__ import absolute_import, division, print_function, unicode_literals;


import tensorflow as tf;
import tensorflow.keras;
import tensorflow_hub as hub;
import tensorflow_datasets as tfds;

import numpy as np;

## Pre-trained text embedding model.

# 20 dims, 2.5% vocabulary converted to OOV buckets.
EMBEDDING_MODEL_20  =   r"https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1";

# 50 dims, ~1M vocabulary size.
EMBEDDING_MODEL_50  =   r"https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1";

# 128 dims, ~1M vocabulary size.
EMBEDDING_MODEL_128 =   r"https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1";


EPOCHS      =   20;
BATCH_SIZE  =   512;

def main():

    ## Import IMDB movie reviews text dataset. 25k dataset total.
    ## Split data: train_ds = 15k examples, val_ds = 10k, test_ds = 25k.
    train_ds, val_ds, test_ds   =   tfds.load(  name="imdb_reviews", 
                                                split=('train[:60%]', 'train[60%:]', 'test'),
                                                as_supervised=True,
                                            );

    ## Testing if dataset format is correct.
    #(X_samples, y_samples) =   next(iter(train_ds.batch(10)));
    #print(  X_samples[0]);
    #print(  y_samples);

    ## Define embedding hub layer for the text model,
    ##  reducing number of features to 50.
    embedding   =   EMBEDDING_MODEL_50;
    hub_layer   =   hub.KerasLayer( embedding,
                                    input_shape = [],
                                    dtype=tf.string,
                                    trainable=True,
                                    );

    ## testing hub_layer on 2 samples, making sure embedding layer works.
    #hub_output  =   hub_layer(  next(iter(  train_ds.batch(2)  )) [0]    );
    #print(hub_output);

    model   =   tf.keras.Sequential(   #name = 'IMDB_sentiment',
                                        [   hub_layer,
                                            #tf.keras.layers.Dropout(0.25),
                                            tf.keras.layers.Dense(64, activation = 'relu'),
                                            tf.keras.layers.BatchNormalization(),

                                            tf.keras.layers.Dense(32, activation = 'relu'),
                                            tf.keras.layers.BatchNormalization(),

                                            tf.keras.layers.Dense(1),
                                            #tf.keras.layers.activations.Softmax,
                                        ]
    );

    model.summary();

    model.compile(  optimizer   =   tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                    loss        =   tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics     =   ['accuracy'],
    );

    #x_train, y_train    =   train_ds.shuffle(15000);
    batched_train_ds    =   train_ds.shuffle(15000).batch(BATCH_SIZE);


    history =   model.fit(  batched_train_ds,
                            #x = ,
                            #y = None,
                            #batch_size = BATCH_SIZE,
                            epochs = EPOCHS,
                            
                            validation_data = val_ds.batch(BATCH_SIZE),     
                            callbacks = None,
                            verbose = 1,
    );

    ## Evaluate model with the test dataset.
    #x_test, y_test  =   test_ds.shuffle(15000);
    print("Test data accuracy result:")
    model.evaluate(test_ds.batch(512), verbose=2);

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    #model.evaluate( test_ds.batch(BATCH_SIZE), verbose=2);


    return 0;


if __name__ == '__main__':  main();
