"""
To run code:
spark-submit minh_tran_competition_test.py Competition/ Competition/yelp_val.csv minh_tran_output.txt
"""
import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
import time

from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import Dataset
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

# Function to read the given csv file
def csvReader(csvFile):
    return csvFile.split(',')

# function to import list data to dict
# input: data in the form of list
# output: dict in the form of {item1: 0, item1: 1...}
def importToDict(listData):
    dictData = {}
    idx = 0
    for item in listData:
        dictData[item] = idx
        idx += 1
    return dictData

# function to import list data to dict ( 2 version of outputs)
# input: data in the form of list
# output 1: dict in the form of {item1: 0, item2: 1...}
# output 2: dict in the form of {0: item1, 1: item2...}
def importToDict2(listData):
    # import user id into a dict {userID: numID} for hashing
    # output is {userID1:0 , userID2:1,...}
    dictData1 = {}
    dictData2 = {}
    idx = 0
    for item in listData:
        dictData1[item] = idx
        dictData2[idx] = item
        idx += 1
    return dictData1, dictData2

def writeFile(outputFilePath, predictions, idUserDict, idBusinessDict):
    out = "user_id, business_id, prediction\n"
    for x in predictions.map(lambda x: (x[0][0], x[0][1], x[1])).collect():
        out += idUserDict[x[0]] + "," + idBusinessDict[x[1]] + "," + str(x[2]) + "\n"
    with open(outputFilePath, "w") as f:
        f.write(out)
        f.close()

def writeTxt(outputFilePath, errorDist, rmse, time):
    # out = "Minh Tran - INF 553 - Competition Description"

    out = "Method Description:\n"
    out += "The method utilizes the hybrid approach from the open-sourced Surprise package. The link to the package is here: https://surprise.readthedocs.io/en/stable/index.html. Two approaches are combined with equal weights. They are baseline method and SVD (Singular value decomposition) (https://surprise.readthedocs.io/en/stable/matrix_factorization.html) \n"
    out += "\nError Distribution:\n"

    keySorted = []
    for key in errorDist.keys():
        # append value of similarity
        keySorted.append(key)
    keySorted = sorted(keySorted)

    for key in keySorted:
        out += key + ": " + str(errorDist[key])+"\n"

    out += "\nRMSE:\n" + str(rmse)+"\n"
    out += "\nExecution Time:\n" + time

    with open(outputFilePath, "w") as f:
        f.write(out)
        f.close()

# function to calculate root mean square error
# input: (user,business), (actual rating, predicted rating)
def calRSME(validation):
    MSE = validation.map(lambda x: (x[1][0] - x[1][1])**2).mean()
    RMSE = pow(MSE,1/2)
    # print("RMSE = ",str(RMSE))
    return RMSE

def calError(validation):
    errorDist = {}
    bin01Key = '>=0 and <1'
    bin12Key = '>=1 and <2'
    bin23Key = '>=2 and <3'
    bin34Key = '>=3 and <4'
    bin45Key = '>=4'

    bin01 = validation.map(lambda x: abs(x[1][0] - x[1][1])).filter(lambda x : x >= 0 and x<1).count()
    bin12 = validation.map(lambda x: abs(x[1][0] - x[1][1])).filter(lambda x: x >= 1 and x < 2).count()
    bin23 = validation.map(lambda x: abs(x[1][0] - x[1][1])).filter(lambda x: x >= 2 and x < 3).count()
    bin34 = validation.map(lambda x: abs(x[1][0] - x[1][1])).filter(lambda x: x >= 3 and x < 4).count()
    bin45 = validation.map(lambda x: abs(x[1][0] - x[1][1])).filter(lambda x: x >= 4).count()

    errorDist[bin01Key] = bin01
    errorDist[bin12Key] = bin12
    errorDist[bin23Key] = bin23
    errorDist[bin34Key] = bin34
    errorDist[bin45Key] = bin45
    return errorDist


"""
MAIN
"""
if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("This function needs 3 input arguments <train_file_name> <test_file_name> <caseID> <outputFilePath>")
        exit(-1)

    # start timer
    start = time.time()

    # import data yelp_train.csv
    trainFileFolder = sys.argv[1]
    trainFilePath  = trainFileFolder + "yelp_train.csv"
    valFilePath    = sys.argv[2]
    # caseID         = int(sys.argv[3])
    outputFilePath = sys.argv[3]

    sc = SparkContext("local[*]")
    sqlContext = SQLContext(sc)

    # import data (user, business, rating)?? how to adapt
    trainRawData = sc.textFile(trainFilePath).filter(lambda x: x != "user_id, business_id, stars").map(
        lambda x: csvReader(x)).map(lambda x: (x[0], x[1], float(x[2]))).persist()

    valRawData = sc.textFile(valFilePath).filter(lambda x: x != "user_id, business_id, stars").map(
        lambda x: csvReader(x)).map(lambda x: (x[0], x[1], float(x[2]))).persist()

    # merge training and validation data:
    data = sc.union([trainRawData, valRawData])
    # print("data count = ",data.count())

    # obtain unique user ID
    userIDList = data.map(lambda x: (x[0], 1)).reduceByKey(lambda old, new: new).map(lambda x: x[0]).collect()
    userIDList.sort()
    # print("len(uni_users) = ",len(userIDList))
    userIDDict, idUserDict  = importToDict2(userIDList)

    # obtain unique business ID {business1:idx1,...} and {idx1:business1,...}
    businessIDList = data.map(lambda x: (x[1], 1)).reduceByKey(lambda old, new: new).map(lambda x: x[0]).collect()
    # print("len(businessIDList)",len(businessIDList))
    businessIDList.sort()
    businessIDDict, idBusinessDict = importToDict2(businessIDList)

    # pre-process data -->  (userIdx, businessIdx, star)
    trainData = trainRawData.map(lambda x: (userIDDict[x[0]], businessIDDict[x[1]], float(x[2]))).persist()

    valData = valRawData.map(lambda x: (userIDDict[x[0]], businessIDDict[x[1]], float(x[2]))).persist()

    valDataToJoin = valRawData.map(lambda x: ((userIDDict[x[0]], businessIDDict[x[1]]), float(x[2]))).persist()

    # building training and validation df
    dfTrain = sqlContext.createDataFrame(trainData, ['user','item','rating']).select("*").toPandas()

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1.0, 5.0))

    # The columns must correspond to user id, item id and ratings (in that order): [['user', 'item', 'rating']]
    trainSet = Dataset.load_from_df(dfTrain[['user','item','rating']], reader)

    # Retrieve the trainset.
    trainset = trainSet.build_full_trainset()

    # Build an algorithm, and train it.
    #algo = KNNWithMeans() # rmse = 1.04
    algo1 = SVD() # rmse = 1.01
    #algo = SVDpp() # rmse = 1.01, 600 sec
    #algo = NMF() # rmse = 1.08
    #algo = SlopeOne() # error: could not serialize object
    #algo = CoClustering() # rmse = 1.05
    # algo= NormalPredictor() # rmse = 1.5
    algo = BaselineOnly() # rmse = 1.005
    # algo = KNNBasic() # rmse = 1.08
    # algo = KNNBaseline() # rmse = 1.033

    algo.fit(trainset)
    algo1.fit(trainset)

    # predict ratings :
    # use 1 algorithm
    # predictions = valData.map(lambda x: ((x[0], x[1]), algo.predict(x[0], x[1], r_ui=x[2], verbose=False).est))

    # combine 2 algorithms equally weighted
    predictions = valData.map(lambda x: ((x[0], x[1]), (algo.predict(x[0], x[1], r_ui=x[2], verbose=False).est+algo1.predict(x[0], x[1], r_ui=x[2], verbose=False).est)/2))
    # print("prediction", predictions.take(5))

    # join estimation with actual result
    validation = predictions.join(valDataToJoin)

    # calculate rmse
    rmse = calRSME(validation)

    # calculate error distribution
    errorDist = calError(validation)
    # print('error: ', errorDist)

    # write output
    writeFile(outputFilePath, predictions, idUserDict, idBusinessDict)

    end = time.time()

    # write text
    writeTxt('minh_tran_description.txt', errorDist, rmse, str(end-start))
