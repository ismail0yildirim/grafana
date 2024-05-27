from minio import Minio
from minio.sse import SseCustomerKey
import urllib3
from minio.error import ResponseError, BucketAlreadyExists, NoSuchBucket, NoSuchKey, AccessDenied
from minio.error import SignatureDoesNotMatch, InvalidBucketError, InvalidAccessKeyId, SignatureDoesNotMatch, \
    BucketAlreadyOwnedByYou
import logging
import os


def init_client(address, first_key, second_key):
    """init connection to Minio via urllib3.PoolManager"""
    httpclient = urllib3.PoolManager(num_pools=50)
    minio_client = Minio(
        address,
        access_key=first_key,
        secret_key=second_key,
        secure=False,
        http_client=httpclient,
    )
    return minio_client


def upload_object_in_bucket(client, bucketname: str, filename: str, filepath, stream: bool, size: int):
    """ Upload object from hard drive or stream as  object name in bucket.
    if file already exists in bucket, add timestamp to name and try it once again

    :param client: on which client to be performed (defined via init_client())
    :param bucketname: name of the bucket, in which the object should stored
    :param filename: name of the file in the bucket
    :param filepath: complete path to file on harddrive
    :param stream: True: if it is a stream object and not located on harddisk uses put(),
    False: for not stream object, uses fput()
    :param size: size of object, needed if object is a stream
    :return:
    """
    # checks if file already exists in bucket
    try:
        client.stat_object(bucketname, filename)

    except NoSuchKey:
        # puts file in bucket, if it doesent exist yet in bucket
        try:
            # stream = object in a stream
            if stream:
                # use put and set ContentType to jpg
                client.put_object(bucketname, filename, filepath, length=size, content_type='image/jpg')
            else:
                client.fput_object(bucketname, filename, filepath)
        except (
                urllib3.exceptions.NewConnectionError,
                urllib3.exceptions.MaxRetryError,
        ) as err:
            logging.debug(err)
        except (ResponseError, ConnectionRefusedError, NoSuchKey, NoSuchBucket) as err:
            logging.debug(err)


def file_exist(client, bucketname: str, filename: str):
    # try to reach file in object, if error file is not existing
    """
    returns True or False, if file under specific name already exists in bucket
    :param client: on which client to be performed (defined via init_client())
    :param bucketname: name of the bucket, in which the object should be checked
    :param filename: name of the file to be checked
    :return: file_exists: True/ False
    """
    try:
        client.stat_object(bucketname, filename)
        return True
    except NoSuchKey:
        return False

# Example
# if __name__ == "__main__":
#     client = init_client()
#     upload_object_in_bucket(client, "20200623171204", "C-G3216940_X1_17_20200406030407_0.jpg", r"D:\Source\C-G3216940_X1_17_20200406030407_0.jpg")
