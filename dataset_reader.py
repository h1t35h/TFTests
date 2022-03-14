import tensorflow as tf
import pandas as pd

class S3DataGenerator(tf.data.Dataset):
    def _generator(s3file: str):
        
        yield s3file


    def __new__(cls, s3file: str):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (), dtype = tf.string),
            args=(s3file,)
        )

def test():
    base = tf.data.Dataset.from_tensor_slices(["file", "file2", "file3"])

    newDs = (
        base.interleave(
            lambda fileName: S3DataGenerator(fileName)
        ).repeat()
    )

    print(list(newDs.take(5).as_numpy_iterator()))

test()