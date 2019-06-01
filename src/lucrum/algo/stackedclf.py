# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import pandas as pd
import xgboost as xgb

###### class to extend sklearn classifiers ################################
class SklearnHelper(object):

    # https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    @property
    def name(self):
        return type(self.clf).__name__

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
###### class to stack (ensemble) sklearn classifiers ######################
class StackedClf():
    def __init__(self, classifiers):
        """Stacked classifer constructor.
        """
        
        # create dictionary for first level classifiers
        clfs = {}
        for i in range(len(classifiers)):
            clfs[classifiers[i].name + "_" + str(i)] = classifiers[i]

        self.clfs = clfs

    @property
    def base_train(self):
        return self._base_train

    def train(self, X_train, y_train, verbose=True):

        self._base_train = pd.DataFrame()

        # loop in classifiers and train models 
        for _, value in self.clfs.items():
            value.train(X_train, y_train)
            self._base_train.insert(self._base_train.shape[1], value.name, value.predict(X_train))

        #self.meta_model = xgb.XGBClassifier(learning_rate = 0.02).fit(self._base_train, y_train)
                                            #learning_rate = 0.02,
                                            # n_estimators= 2000,
                                            # max_depth= 4,
                                            # min_child_weight= 2,
                                            # #gamma=1,
                                            # gamma=0.9,                        
                                            # subsample=0.8,
                                            # colsample_bytree=0.8,
                                            # objective= 'binary:logistic',
                                            # nthread= -1,
                                            # scale_pos_weight=1).fit(base_train.values, y_train)

        if verbose == True:
            print("Training complete") 
        
    def predict(self, X_test):
        base_predict = pd.DataFrame()
        for _, value in self.clfs.items():
            base_predict.insert(base_predict.shape[1], value.name, value.predict(X_test))

        return self.meta_model.predict(base_predict.values)

    def predict_base(self, X_test):
        base_predictions = []
        for _, value in self.clfs.items():
            base_predictions.append((value.name, value.predict(X_test)))
        return base_predictions
    # def feature_importance(self):
    #     importance = []
    #     for _, value in self.clfs.items():
    #         importance.append((value.name, value.feature_importances))

    #     return importance
