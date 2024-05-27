# import boto3
# from botocore.client import Config
import predict
import os
import glob

def loadNewestVersionImgClass(client, bucket, boardside, boardtype, modelType, listofBoards, listofSides):
    if modelType.startswith("ImgClass") == True:

        boardtype = str(boardtype)
        boardside= str(boardside)
        modelType = str(modelType)
        folder = (boardtype + '/' + modelType + '/' + boardside)
        modelBucket = client.Bucket(bucket)
        newestVersions = [f.key.split(folder + "/")[1] for f in modelBucket.objects.filter(Prefix=folder).all()][-2:]
        version = ((newestVersions[0]).split('.'))[0]

        if listofBoards[-2] != boardtype or (boardside not in listofSides) == True:
            if listofBoards[-2] != boardtype or listofBoards[-2] == 'spacer':
                listofSides.clear()
            global model
            if len(os.listdir('ModelCache')) > 0 and listofBoards[-2] != boardtype:
                files = glob.glob('ModelCache/*')
                for f in files:
                    os.remove(f)

            for name in newestVersions:
                client.Bucket(bucket).download_file((boardtype + '/' + modelType + '/' + boardside + '/' + name),
                                                    ('ModelCache/' + (boardtype + '_' + boardside + name)))

            model = predict.PredictionService().init_model(os.path.join('ModelCache', (boardtype + '_' + boardside + newestVersions[0])),
                                       os.path.join('ModelCache', (boardtype + '_' + boardside + newestVersions[-1])))

            listofSides.append(boardside)
            print('Initialized model:', boardtype, boardside)
            del listofBoards[0]
            return model, version

        elif boardtype == listofBoards[-2] and any(boardside in s for s in listofSides) and listofSides[-1] != boardside:
            model = predict.PredictionService().init_model(os.path.join('ModelCache', (boardtype + '_' + boardside + newestVersions[0])),
                                       os.path.join('ModelCache', (boardtype + '_' + boardside + newestVersions[-1])))
            listofSides.append(boardside)
            del listofBoards[0]
            print('Used already downloaded model', boardside, boardtype)
            return model, version

        elif boardtype == listofBoards[-2] and listofSides[-1] == boardside:
            print('Used already initialized Model: ', boardside, boardtype)
            listofSides.append(boardside)
            del listofBoards[0]
            return model, version

        else:
            print('Error: Something is not right with the bucket or the filename, or for the current product is no model available')
            return model, version

    else:
        print('No model for this model type in the bucket. A change of the selected model type in the config.txt might be needed')
        return model, version


def loadNewestVersionAutoencoder(client, bucket, boardside, boardtype, modelType, listofBoards, listofSides):
    if modelType.startswith("AE") == True:

        boardtype = str(boardtype)
        boardside= str(boardside)
        modelType = str(modelType)
        folder = (boardtype + '/' + modelType + '/' + boardside)
        modelBucket = client.Bucket(bucket)
        newestVersions = [f.key.split(folder + "/")[1] for f in modelBucket.objects.filter(Prefix=folder).all()][-2:]
        if len(newestVersions) > 1:
            version = ((newestVersions[0]).split('.'))[0]

            if listofBoards[-2] != boardtype or (boardside not in listofSides) == True:
                if listofBoards[-2] != boardtype or listofBoards[-2] == 'spacer':
                    listofSides.clear()
                global autoencoder
                if len(os.listdir('ModelCacheAE')) > 0 and listofBoards[-2] != boardtype:
                    files = glob.glob('ModelCacheAE/*')
                    for f in files:
                        os.remove(f)

                for name in newestVersions:
                    client.Bucket(bucket).download_file((boardtype + '/' + modelType + '/' + boardside + '/' + name),
                                                        ('ModelCacheAE/' + (boardtype + '_' + boardside + name)))

                autoencoder = predict.PredictionService().init_autoencoder(os.path.join('ModelCacheAE', (boardtype + '_' + boardside + newestVersions[0])),
                                        os.path.join('ModelCacheAE', (boardtype + '_' + boardside + newestVersions[-1])))

                listofSides.append(boardside)
                print('Initialized Autoencoder:', boardtype, boardside)
                del listofBoards[0]
                return autoencoder, version

            elif boardtype == listofBoards[-2] and any(boardside in s for s in listofSides) and listofSides[-1] != boardside:
                autoencoder = predict.PredictionService().init_autoencoder(os.path.join('ModelCacheAE', (boardtype + '_' + boardside + newestVersions[0])),
                                        os.path.join('ModelCacheAE', (boardtype + '_' + boardside + newestVersions[-1])))
                listofSides.append(boardside)
                del listofBoards[0]
                print('Used already downloaded Autoencoder', boardside, boardtype)
                return autoencoder, version

            elif boardtype == listofBoards[-2] and listofSides[-1] == boardside:
                print('Used already initialized Autoencoder: ', boardside, boardtype)
                listofSides.append(boardside)
                del listofBoards[0]
                return autoencoder, version

            else:
                print('Error: Something is not right with the bucket or the filename, or for the current product is no autoencoder available')
                return autoencoder, version
        else:
            print('No autoencoder for this model type in the bucket.')
            version = None
            return 0, version

    else:
        print('No autoencoder for this model type in the bucket. A change of the selected autoencoder model type in the config.txt might be needed')
        return autoencoder, version