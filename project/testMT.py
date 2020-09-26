from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
Here is a simple example showing how you can (down)load a dataset, split it for 5-fold cross-validation, and compute the MAE and RMSE of the SVD algorithm.
"""
#
# from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection import cross_validate
#
#
# # Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')
#
# # Use the famous SVD algorithm.
# algo = SVD()
#
# # Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

"""
If you don’t want to run a full cross-validation procedure, you can use the train_test_split() to sample a trainset and a testset with given sizes, and use the accuracy metric of your chosing. You’ll need to use the fit() method which will train the algorithm on the trainset, and the test() method which will return the predictions made from the testset:
"""
# from surprise import SVD
# from surprise import Dataset
# from surprise import accuracy
# from surprise.model_selection import train_test_split
#
#
# # Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('ml-100k')
#
# # sample random trainset and testset
# # test set is made of 25% of the ratings.
# trainset, testset = train_test_split(data, test_size=.25)
#
# # We'll use the famous SVD algorithm.
# algo = SVD()
#
# # Train the algorithm on the trainset, and predict ratings for the testset
# algo.fit(trainset)
# predictions = algo.test(testset)
#
# # you can train and test an algo with 1 line
# # predictions = algo.fit(trainset).test(testset)
#
# # Then compute RMSE
# accuracy.rmse(predictions)

"""
Train on a whole trainset and the predict() method
Obviously, we could also simply fit our algorithm to the whole dataset, rather than running cross-validation. This can be done by using the build_full_trainset() method which will build a trainset object:
"""
from surprise import KNNBasic
from surprise import Dataset

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo = KNNBasic()
algo.fit(trainset)

# predict ratings : Let’s say you’re interested in user 196 and item 302 (make sure they’re in the trainset!), and you know that the true rating rui=4:
uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
print(type(pred.est))

"""
Use a custom dataset
Surprise has a set of builtin datasets, but you can of course use a custom dataset. Loading a rating dataset can be done either from a file (e.g. a csv file), or from a pandas dataframe. Either way, you will need to define a Reader object for Surprise to be able to parse the file or the dataframe.
"""

# # To load a dataset from a file (e.g. a csv file), you will need the load_from_file() method:
# import os
# from surprise import BaselineOnly
# from surprise import Dataset
# from surprise import Reader
# from surprise.model_selection import cross_validate
#
# # path to dataset file
# file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
#
# # As we're loading a custom dataset, we need to define a reader. In the
# # movielens-100k dataset, each line has the following format:
# # 'user item rating timestamp', separated by '\t' characters.
# reader = Reader(line_format='user item rating timestamp', sep='\t')
#
# data = Dataset.load_from_file(file_path, reader=reader)
#
# # We can now use this dataset as we please, e.g. calling cross_validate
# cross_validate(BaselineOnly(), data, verbose=True)

"""
Using prediction algo: Baseline estimate configuration
Baselines can be estimated in two different ways:
"""
# # Using Alternating Least Squares (ALS).
# print('Using ALS')
# bsl_options = {'method': 'als',
#                'n_epochs': 5,
#                'reg_u': 12,
#                'reg_i': 5
#                }
# algo = BaselineOnly(bsl_options=bsl_options)
#
# # Using Stochastic Gradient Descent (SGD).
# print('Using SGD')
# bsl_options = {'method': 'sgd',
#                'learning_rate': .00005,
#                }
# algo = BaselineOnly(bsl_options=bsl_options)
#
# # similarity measures may use baselines
# bsl_options = {'method': 'als',
#                'n_epochs': 20,
#                }
# sim_options = {'name': 'pearson_baseline'}
# algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)
#
# """
# Similarity measure configuration
# """
#
# sim_options = {'name': 'cosine',
#                'user_based': False  # compute  similarities between items
#                }
# algo = KNNBasic(sim_options=sim_options)

"""
BENCHMARK EXAMPLES
"""
'''This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file'''

# import time
# import datetime
# import random
#
# import numpy as np
# import six
# from tabulate import tabulate
#
# from surprise import Dataset
# from surprise.model_selection import cross_validate
# from surprise.model_selection import KFold
# from surprise import NormalPredictor
# from surprise import BaselineOnly
# from surprise import KNNBasic
# from surprise import KNNWithMeans
# from surprise import KNNBaseline
# from surprise import SVD
# from surprise import SVDpp
# from surprise import NMF
# from surprise import SlopeOne
# from surprise import CoClustering
#
# # The algorithms to cross-validate
# classes = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
#            CoClustering, BaselineOnly, NormalPredictor)
#
# # ugly dict to map algo names and datasets to their markdown links in the table
# stable = 'http://surprise.readthedocs.io/en/stable/'
# LINK = {'SVD': '[{}]({})'.format('SVD',
#                                  stable +
#                                  'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD'),
#         'SVDpp': '[{}]({})'.format('SVD++',
#                                    stable +
#                                    'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp'),
#         'NMF': '[{}]({})'.format('NMF',
#                                  stable +
#                                  'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF'),
#         'SlopeOne': '[{}]({})'.format('Slope One',
#                                       stable +
#                                       'slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne'),
#         'KNNBasic': '[{}]({})'.format('k-NN',
#                                       stable +
#                                       'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic'),
#         'KNNWithMeans': '[{}]({})'.format('Centered k-NN',
#                                           stable +
#                                           'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans'),
#         'KNNBaseline': '[{}]({})'.format('k-NN Baseline',
#                                          stable +
#                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline'),
#         'CoClustering': '[{}]({})'.format('Co-Clustering',
#                                           stable +
#                                           'co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering'),
#         'BaselineOnly': '[{}]({})'.format('Baseline',
#                                           stable +
#                                           'basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly'),
#         'NormalPredictor': '[{}]({})'.format('Random',
#                                              stable +
#                                              'basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor'),
#         'ml-100k': '[{}]({})'.format('Movielens 100k',
#                                      'http://grouplens.org/datasets/movielens/100k'),
#         'ml-1m': '[{}]({})'.format('Movielens 1M',
#                                    'http://grouplens.org/datasets/movielens/1m'),
#         }
#
#
# # set RNG
# np.random.seed(0)
# random.seed(0)
#
# # specify dataset
# dataset = 'ml-100k'
# data = Dataset.load_builtin(dataset)
#
# kf = KFold(random_state=0)  # folds will be the same for all algorithms.
#
# table = []
# for klass in classes:
#     start = time.time()
#     out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
#     cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
#     link = LINK[klass.__name__]
#     mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
#     mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))
#
#     new_line = [link, mean_rmse, mean_mae, cv_time]
#     print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
#     table.append(new_line)
#
# header = [LINK[dataset],
#           'RMSE',
#           'MAE',
#           'Time'
#           ]
# print(tabulate(table, header, tablefmt="pipe"))