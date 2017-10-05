import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import os.path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.neighbors import KernelDensity
import datetime

'''
Random useful functions
'''

def scoreAUC(y,probs):
    ps = np.linspace(0.,1.,num=100)
    prs = []
    nrs = []
    for p in ps:
        preds = probs[:,0]<p
        pr = np.sum((y & preds))/float(np.sum(y))
        nr = np.sum((1-y & 1-preds))/float(np.sum(1-y))
        nrs.append(nr)
        prs.append(pr)
    xs = 1-np.array(nrs)
    ys = np.array(prs)
    dxs = xs[1:] - xs[:-1]
    ays = .5*(ys[1:] + ys[:-1])
    auc = np.sum(ays*dxs)
    return {'score':auc,'fpr':xs,'tpr':ys}



'''
Pre-process data
'''



def convertColumnToWeeks(col):
    return np.floor((pd.to_datetime(col) - datetime.datetime(2007,01,01)).dt.days/7.).astype(int)

'''

Feature engineering functions

'''

def addDateCol(df):
    '''
    add leaky feature to column
    '''
    print df.groupby('Date').count().columns
    dt_count = df.groupby('Date').count()[['Address']] #get mutiple counts on a certain date
    dt_count.columns = ['DateCount'] #create new column for how many times date repeated.
    test = pd.merge(df, dt_count, how='inner', left_on='Date', right_index=True)
    return test['DateCount']

def wnvPresentFeature(df_train,df):
    '''
    Take in df_train to train kde and output column
    in df
    returns df with new column
    '''
    Xs = df_train[df_train['WnvPresent']==1][['Longitude','Latitude']].get_values() #take only values where WNV present.
    kde = KernelDensity(bandwidth=0.02,
                        kernel='gaussian')
    kde.fit(Xs)


    Xs = df[['Longitude','Latitude']].get_values()
    df['pWNV1'] = np.exp(kde.score_samples(Xs))

    return df

def mosquitoPresenceFeature(train,test):
    pMos_train = np.zeros(len(train.index))
    pMos_test = np.zeros(len(test.index))
    for week in train['week'].unique():
        print("Week: {}".format(week))
        inds = train['week']==week
        X = train[inds][['Longitude','Latitude']].get_values()
        y = train[inds]['NumMosquitos'].get_values()
        gpr = GaussianProcessRegressor(kernel=kernels.Matern(length_scale=0.1))
        gpr.fit(X,y)

        pMos_train[inds] = gpr.predict(X)

        #now use model to predict on test data
        inds = test['week']==week
        X = test[inds][['Longitude','Latitude']].get_values()
        if X.size > 0:
            pMos_test[inds] = gpr.predict(X)

    train['pMos1'] = pMos_train
    test['pMos1'] = pMos_test
    return train,test



'''

Load in data

'''

def loadPreData(features=False):
    """
        Beating the Benchmark
        West Nile Virus Prediction @ Kaggle
        __author__ : Abhihsek
    """
    # Load dataset
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    sample = pd.read_csv('./data/sampleSubmission.csv')
    weather = pd.read_csv('./data/weather.csv')

    # Get labels
    labels = train.WnvPresent.values


    #add leaky feature
    train_datecount = addDateCol(train)
    test_datecount = addDateCol(test)

    # Not using codesum for this benchmark
    weather = weather.drop('CodeSum', axis=1)

    # Split station 1 and 2 and join horizontally
    weather_stn1 = weather[weather['Station']==1]
    weather_stn2 = weather[weather['Station']==2]
    weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn2 = weather_stn2.drop('Station', axis=1)
    weather = weather_stn1.merge(weather_stn2, on='Date')

    # replace some missing values and T with -1
    weather = weather.replace('M', -1)
    weather = weather.replace('-', -1)
    weather = weather.replace('T', -1)
    weather = weather.replace(' T', -1)
    weather = weather.replace('  T', -1)

    # Functions to extract month and day from dataset
    # You can also use parse_dates of Pandas.
    def create_month(x):
        return x.split('-')[1]

    def create_day(x):
        return x.split('-')[2]

    train['month'] = train.Date.apply(create_month)
    train['day'] = train.Date.apply(create_day)
    test['month'] = test.Date.apply(create_month)
    test['day'] = test.Date.apply(create_day)

    # Add integer latitude/longitude columns
    train['Lat_int'] = train.Latitude.apply(int)
    train['Long_int'] = train.Longitude.apply(int)
    test['Lat_int'] = test.Latitude.apply(int)
    test['Long_int'] = test.Longitude.apply(int)

    # drop address columns
    train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
    test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

    # Merge with weather data
    train = train.merge(weather, on='Date')
    test = test.merge(weather, on='Date')
    train = train.drop(['Date'], axis = 1)
    test = test.drop(['Date'], axis = 1)

    # Convert categorical data to numbers
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train['Species'].values) + list(test['Species'].values))
    train['Species'] = lbl.transform(train['Species'].values)
    test['Species'] = lbl.transform(test['Species'].values)

    lbl.fit(list(train['Street'].values) + list(test['Street'].values))
    train['Street'] = lbl.transform(train['Street'].values)
    test['Street'] = lbl.transform(test['Street'].values)

    lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
    train['Trap'] = lbl.transform(train['Trap'].values)
    test['Trap'] = lbl.transform(test['Trap'].values)

    # drop columns with -1s
    train = train.ix[:,(train != -1).any(axis=0)]
    test = test.ix[:,(test != -1).any(axis=0)]

    #add leaky feature
    train['DateCount'] = train_datecount
    test['DateCount'] = test_datecount

    if features:
        train,test = loadFeatures(train,test)



    return {'train':train,'test':test,'labels':labels}


def generateFeatures():
    #load in data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    #create weeks since start of measurements column.
    train['weeks_since_start'] = convertColumnToWeeks(train['Date'])
    test['weeks_since_start'] = convertColumnToWeeks(test['Date'])
    train['week'] = pd.to_datetime(train['Date']).dt.week
    test['week'] = pd.to_datetime(test['Date']).dt.week

    test =  wnvPresentFeature(train,test)
    train = wnvPresentFeature(train,train)

    train,test = mosquitoPresenceFeature(train,test)

    train = train.drop(['weeks_since_start'], axis = 1)
    test = test.drop(['weeks_since_start'], axis = 1)

    train.to_csv('./data/train_features.csv')
    test.to_csv('./data/test_features.csv')

def loadFeatures(train,test):
    feature_test_path = './data/test_features.csv'
    feature_train_path = './data/train_features.csv'

    if not (os.path.isfile(feature_test_path) and os.path.isfile(feature_train_path)):
        print('Features not generated yet.');
        print('Generating features. Might take some time...');
        generateFeatures()


    trainf = pd.read_csv('./data/train_features.csv')
    testf = pd.read_csv('./data/test_features.csv')

    test['pWNV1'] = testf['pWNV1']
    train['pWNV1'] = trainf['pWNV1']

    test['pMos1'] = testf['pMos1']
    train['pMos1'] = trainf['pMos1']

    return train,test
