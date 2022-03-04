import tensorflow as tf
import boto3
import pandas as pd

def getFileLists(bucket: str, prefix: str):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket)
        return list(s3object.key for s3object in bucket.objects.filter(Prefix=prefix))


def pandas_to_dataset(pandas_ds) -> tf.data.Dataset:
    return None

def pandasGenerator(s3bucket: str, s3prefix: str):
    s3files = getFileLists(s3bucket, s3prefix)
    for s3file in s3files:
        panda_ds = pd.read_parquet(s3file)
        dataset = pandas_to_dataset(pandas_ds)
        for data in dataset:
            yield data

def createS3Dataset(s3bucket:str, s3prefix: str) -> tf.data.Dataset :
    return tf.data.Dataset.from_generator(pandasGenerator(s3bucket, s3prefix))


def main():
    print("Hello World!")