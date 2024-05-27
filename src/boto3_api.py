import boto3
from botocore.client import Config


def init_s3_client(url, access_key, secret_key):
    url = 'http://' + url
    s3_client = boto3.resource('s3',
                               endpoint_url=url,
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key,
                               config=Config(signature_version='s3v4'),
                               region_name='eu-central-1'
                               )
    return s3_client


def download_s3(client, bucketName, filename, workingDirectory):
    client.Bucket(bucketName).download_file(filename, workingDirectory)

def upload_s3(client, path, filename, bucketName):
    client.Bucket(bucketName).upload_file(path, filename)

def getModelsAndSides(client, bucketName):
    modelBucket = client.Bucket(bucketName)
    availableSides = list(set([f.key.split("/")[2]  for f in modelBucket.objects.filter().all()]))
    dic = {}
    for product in list(set([f.key.split("/")[0] for f in modelBucket.objects.filter().all()])):
        dic[product] = []
        for f in modelBucket.objects.filter(Prefix=product):
            f = f.key.split(product + "/")[1]
            entry = f.split('/')[0] + f.split('/')[1]
            dic[product].append(entry)

        dic[product] = list(set(dic[product]))
    return dic, availableSides

# def upload_awsS3(client, path, filename, bucketName):
#     client.upload_file(path, bucketName, filename)
    #print(response)