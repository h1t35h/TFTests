import tensorflow as tf
import pandas as pd
import random
import string
import time
from tensorflow import feature_column
from tensorflow.keras import layers
import os

def uniqueValues(valuesCount: int, string_length : int):
    values = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = string_length))
                             for i in range(valuesCount)]

    values.append("ABC")
    values = list(set(values))
    print(f"Total Vocab: {len(values)}")
    return values

def _df_to_dataset(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    labels = dataframe.pop('labels')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    return ds

def createFeatureColumns():
    feature_columns = []
    indicator_columns = ["embedded_col","rand_col"]
    for colm in indicator_columns:
        feature_col = feature_column.categorical_column_with_vocabulary_list(
            colm, uniqueValues(999999, 5))
        indicator_column = feature_column.embedding_column(feature_col, dimension=10)
        feature_columns.append(indicator_column)
    return feature_columns


def dfGenerator():
    d = {
        'embedded_col': ["ABC", "DEF", "ABCF", "DEFG"], 
        'rand_col': ["Test", "doubleTest", "ABC", "DEF"],
        'random_col': [3, 4, 5 , 5], 
        'labels':[1, 2, 3 ,4]}

    df = pd.DataFrame(data=d)
    
    for data in df.itertuples(index=False):
        dt = dict(data._asdict())
        label = dt.pop('labels')
        yield dt, label
    return

def createDataset():
    return tf.data.Dataset.from_generator(dfGenerator, 
    output_signature =(
        {
            'embedded_col': tf.TensorSpec(shape=(), dtype=tf.string),
            'rand_col': tf.TensorSpec(shape=(), dtype=tf.string),
            'random_col': tf.TensorSpec(shape=(), dtype=tf.int64)
        }, tf.TensorSpec(shape=(), dtype=tf.int64)
    ))

def demo(example_ds):
    feature_cols = createFeatureColumns()
    demo_fl= tf.keras.layers.DenseFeatures(feature_cols)
    print(demo_fl(example_ds))

def process_data():
    dataset = tf.data.Dataset.from_generator(dfGenerator, 
    output_signature = ({
        'embedded_col': tf.TensorSpec(shape=(), dtype=tf.string),
        'rand_col': tf.TensorSpec(shape=(), dtype=tf.string),
        'random_col': tf.TensorSpec(shape=(), dtype=tf.int64)
    }, tf.TensorSpec(shape=(), dtype=tf.int64)))
    dataset = dataset.repeat().batch(2, drop_remainder=False)
    os.system("clear")
    example = next(iter(dataset))[0]
    demo(example_ds=example)

def _feature_columns():
    feature_colmns = []
    for colmn in ['embedded_col','rand_col']:
        feature_col = (feature_column
                    .categorical_column_with_vocabulary_list(colmn, ["ABC", "DEF", "ABCF", "DEFG",
                    "Test", "doubleTest"]))
        feature_colmns.append(feature_column.indicator_column(feature_col))

    feature_colmns.append(feature_column.numeric_column('random_col'))
    return feature_colmns

def testModel(dataset):
    feature_columns = _feature_columns()
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(1280, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1280, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer ='adam',
                    loss =tf.keras.losses.MeanAbsoluteError(),
                    metrics=['accuracy', 'mean_absolute_error'])

    model.fit(dataset, epochs=10, verbose=1)


def testDiff():
    d = {
        'embedded_col': ["ABC", "DEF", "ABCF", "DEFG"], 
        'rand_col': ["Test", "doubleTest", "ABC", "DEF"],
        'random_col': [3, 4, 5 , 5], 
        'labels':[1, 2, 3 ,4]}
    df = pd.DataFrame(data=d)
    labels = df.pop('labels')
    normal_ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)).batch(2)
    testModel(normal_ds)

    gen_ds = createDataset().batch(2)
    testModel(gen_ds)



testDiff()