#coding=utf-8
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import cPickle
from math import ceil

from sklearn.cross_validation import KFold

def make_yuce():
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')

    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']}) #将来要写入的文件

    # xgb
    import xgboost as xgb
    param = {'bst:max_depth':6,
             'bst:eta':0.1,
             'objective':'reg:linear',
             'gamma':0,
             'silent':1,
             'colsample_bytree':0.8,
             'subsample' : 0.8,
             'min_child_weight': 15,
             # 'alpha':10
            'lambda':10
             # ''
             }

    # 两个都是0.8 -》 176.765878
    # 176.569872  176.400838
    # 'lambda':10  -> 172.486285
    # 60次迭代 ： 171.844895
    # 50 171.213232
    #30 170.484217
    plst = param.items()

    '''只用月总销售值的'''
    # col = list(test.columns[1:])
    # for i in range(1,12):
    #     col.remove('div_' + str(i))
    '''只用月平均销售值的'''
    # col = list(test.columns[1:])
    # for i in range(1,12):
    #     col.remove('month_price_zong_' + str(i))
    '''都用'''
    col = list(test.columns[1:])

    for i in range(0,12):
        print('begin to pred which label ? ' , 'label_' + str(i))
        # 特征列
        one_line_train = col
        one_line_test = col
        # label列
        lie = 'num_' + str(i + 12)
        # 开xgb模型
        feature_train = train[one_line_train].as_matrix() #特征列
        bizhi_lie = train[lie]

        dtrain = xgb.DMatrix( feature_train, bizhi_lie)   #训练集的所有特征列，训练集的“要预测的那一列”

        # # 打印auc的代码，还别说，挺高的，是从 test-auc 从 0.7 开始上升
        # bst = xgb.cv(plst, dtrain, num_boost_round=100 , nfold=5 , metrics ='auc',verbose_eval=True )

        # # 将 bst 的列名，全部变化，也是为了区分每一个label的 AUC
        # col = list(bst.columns)
        # for k in range(4):
        #     col[k] = 'label_' + str(i) + '_' + col[k]
        # bst.columns = col
        # bst.to_csv('../data/model/every_model_auc.csv',index=False,mode='a')

        dtest = xgb.DMatrix(test[one_line_test].as_matrix()) #测试集的所有特征列

        bst = xgb.train(plst , dtrain , 30)

        Prediction = bst.predict(dtest) # 测试集的所有特征列
        Prediction = DataFrame(Prediction)
        Prediction = Prediction[Prediction.columns[0]]

        Submission14[lie] = Prediction.astype('int')

    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])

    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv' , index = False , encoding="utf-8" , mode='a')

def xgb_kf():
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')
    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']})

    # xgb
    import xgboost as xgb
    param = {'bst:max_depth': 6,
             'bst:eta': 0.1,
             'objective': 'reg:linear',
             'gamma': 0,
             'silent': 1,
             'colsample_bytree': 0.8,
             'subsample': 0.8,
             'min_child_weight': 15,
             # 'alpha':10
             'lambda': 10
             # ''
             }
    plst = param.items()

    '''都用'''
    fea_all = list(test.columns[1:])


    for i in range(0,12):
        print('begin to pred which label ? ' , 'label_' + str(i))

        train_data = train[fea_all]
        train_data = np.array(train_data)  # 训练集（特征列）
        train_target = train['num_' + str(i + 12)]
        train_target = np.array(train_target)  # 训练集（label列）

        num_fold = 0
        random_state = 51
        models = []
        nfolds = 18
        kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)

        for train_index, test_index in kf:  # train_index test_index 在 5次循环中每一次都不同，但是每一次的这两个list合起来刚好是 0~len(train_data)中的每一个数字
            print(u'which model begin to make : ', num_fold)
            num_fold += 1

            # n_estimators=1)
            X_train = train_data[train_index]
            X_train = DataFrame(X_train)  # 训练集的“特征列”（DataFrame类型）
            Y_train = train_target[train_index]
            Y_train = DataFrame(Y_train)
            Y_train = Y_train[0]  # 训练集的“label列”（Series类型）

            X_valid = train_data[test_index]
            X_valid = DataFrame(X_valid)  # 测试集的“特征列”（DataFrame类型）
            Y_valid = train_target[test_index]
            Y_valid = DataFrame(Y_valid)
            Y_valid = Y_valid[0]  # 测试集的“label列”（DataFrame类型）

            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))
            print('begin to make model')
            dtrain = xgb.DMatrix(X_train.as_matrix(), Y_train)
            bst = xgb.train(plst, dtrain, 30)

            print('make model over')

            models.append(bst)  # 将上面刚刚训练好的模型存储起来

        # 预测
        num_fold = 0
        yfull_test = []
        for j in range(nfolds):
            print(u'which model we will use to predicte : ', num_fold)
            model = models[j]  # 先拿第一个模型来搞
            num_fold += 1
            dtest = xgb.DMatrix(test[fea_all].as_matrix())
            y_pred = model.predict(dtest)

            yfull_test.append(list(y_pred))

        a = np.array(yfull_test[0])  # 第一个模型的预测结果
        for j in range(1, nfolds):
            a += np.array(yfull_test[j])
        a /= nfolds


        Submission14['num_' + str(i + 12)] = a


    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])

    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv' , index = False , encoding="utf-8" , mode='a')


def make_gbdt():
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    from sklearn.ensemble import GradientBoostingRegressor
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')

    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']}) #将来要写入的文件

    original_params = {'n_estimators': 50, 'max_leaf_nodes': 4, 'max_depth': 6, 'random_state': 2,
                       'learning_rate': 0.1, 'min_samples_leaf': 12,
                       'min_samples_split': 5, 'subsample': 0.8}
    params = dict(original_params)

    '''都用'''
    col = list(test.columns[1:])

    for i in range(0, 12):
        gbdt = GradientBoostingRegressor(**params)
        print('begin to pred which label ? ', 'label_' + str(i))
        # 特征列
        one_line_train = col
        one_line_test = col
        # label列
        lie = 'num_' + str(i + 12)
        # 开gbdt模型
        feature_train = train[one_line_train]  # 训练集的“特征集合”
        bizhi_lie = train[lie]  # 训练集的“label列”

        print('begin to build model')
        gbdt.fit(feature_train.as_matrix(), bizhi_lie.as_matrix())  # 训练出来模型
        print('build model over')

        prediction = gbdt.predict(test[one_line_test].as_matrix())
        prediction = DataFrame(prediction)
        prediction = prediction[prediction.columns[0]]

        Submission14[lie] = prediction.astype('int')

    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])

    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv', index=False, encoding="utf-8", mode='a')

def gbdt_kf():
    from sklearn.ensemble import GradientBoostingRegressor
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')
    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']})

    original_params = {'n_estimators': 50, 'max_leaf_nodes': 4, 'max_depth': 6, 'random_state': 2,
                       'learning_rate': 0.1, 'min_samples_leaf': 12,
                       'min_samples_split': 5, 'subsample': 0.8}
    params = dict(original_params)

    '''都用'''
    fea_all = list(test.columns[1:])


    for i in range(0,12):
        print('begin to pred which label ? ' , 'label_' + str(i))
        gbdt = GradientBoostingRegressor(**params)
        train_data = train[fea_all]
        train_data = np.array(train_data)  # 训练集（特征列）
        train_target = train['num_' + str(i + 12)]
        train_target = np.array(train_target)  # 训练集（label列）

        num_fold = 0
        random_state = 51
        models = []
        nfolds = 8
        kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)

        for train_index, test_index in kf:  # train_index test_index 在 5次循环中每一次都不同，但是每一次的这两个list合起来刚好是 0~len(train_data)中的每一个数字
            print(u'which model begin to make : ', num_fold)
            num_fold += 1

            # n_estimators=1)
            X_train = train_data[train_index]
            X_train = DataFrame(X_train)  # 训练集的“特征列”（DataFrame类型）
            Y_train = train_target[train_index]
            Y_train = DataFrame(Y_train)
            Y_train = Y_train[0]  # 训练集的“label列”（Series类型）

            X_valid = train_data[test_index]
            X_valid = DataFrame(X_valid)  # 测试集的“特征列”（DataFrame类型）
            Y_valid = train_target[test_index]
            Y_valid = DataFrame(Y_valid)
            Y_valid = Y_valid[0]  # 测试集的“label列”（DataFrame类型）

            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))
            print('begin to make model')
            gbdt.fit(X_train.as_matrix(), Y_train.as_matrix())
            print('make model over')

            models.append(gbdt)  # 将上面刚刚训练好的模型存储起来

        # 预测
        num_fold = 0
        yfull_test = []
        for j in range(nfolds):
            print(u'which model we will use to predicte : ', num_fold)
            model = models[j]  # 先拿第一个模型来搞
            num_fold += 1
            y_pred = model.predict(test[fea_all].as_matrix())

            yfull_test.append(list(y_pred))

        a = np.array(yfull_test[0])  # 第一个模型的预测结果
        for j in range(1, nfolds):
            a += np.array(yfull_test[j])
        a /= nfolds


        Submission14['num_' + str(i + 12)] = a


    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])
        Submission14[i] = Submission14[i].apply(lambda x : round(x,2))
    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv' , index = False , encoding="utf-8" , mode='a')


def yuce_2_tijiao():
    # 首先将刚刚形成的那个文件的格式进行个转换
    yuce_1 = pd.read_csv('../data/yuce.csv')

    list_month = ['2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01',
                  '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01', '2017-01-01']
    # 调整后的列名
    col_reset = ['product_id', 'product_month', 'ciiquantity_month']
    col = []
    for i in range(12, 26):
        col.append('num_' + str(i))

    for i in yuce_1['product_id'].unique():
        # print(i)
        data = yuce_1[yuce_1['product_id'] == i]
        data = data.reset_index(drop=True)

        new = DataFrame()
        new['product_month'] = list(list_month)
        new['product_id'] = data['product_id'].unique()[0]
        new['ciiquantity_month'] = list(data[col].T[0])
        # ciiquantity_month 整形的变化、列名的调整
        new['ciiquantity_month'] = new['ciiquantity_month'].astype('int')
        new['ciiquantity_month'] = abs(new['ciiquantity_month'])
        new = new[col_reset]

        new.to_csv('../data/tijiao_part1.csv', index=False, mode='a', encoding='utf-8', header=None)

# 这个函数的目的是：将tijiao_part1.csv其余的id的提交数据
def try_1():
    yuce_1 = pd.read_csv('../data/tijiao_part1.csv', header=None)
    yuce_1.columns = ['product_id','product_month','ciiquantity_month']
    yuce_3476 = yuce_1['product_id'].unique()
    yuce_4000 = range(1,4001)
    chaji = list(set(yuce_4000).difference(set(yuce_3476))) # a中有而b中没有的
    for i in chaji:
        # print(i)
        data = yuce_1[yuce_1['product_id'] == 1]
        data = data.reset_index(drop=True)

        data['product_id'] = i
        data['ciiquantity_month'] = 0
        data = data[['product_id','product_month','ciiquantity_month']]

        data.to_csv('../data/tijiao_part2.csv', index=False, mode='a', encoding='utf-8', header=None)
def try_1_1():
    tijiao_part1 = pd.read_csv('../data/tijiao_part1.csv', header=None)
    tijiao_part2 = pd.read_csv('../data/tijiao_part2.csv', header=None)
    tijiao_part1.columns = ['product_id','product_month','ciiquantity_month']
    tijiao_part2.columns = ['product_id', 'product_month', 'ciiquantity_month']
    tijiao_part = tijiao_part1.append(tijiao_part2)
    tijiao_part = tijiao_part.sort_values(by=['product_id','product_month'])
    for i in ['product_id','ciiquantity_month']:
        tijiao_part[i] = tijiao_part[i].astype('int')
    tijiao_part.to_csv('../data/tijiao_part.csv', index=False, mode='a', encoding='utf-8')


# 这套函数要解决的问题是：
def try_2():
    yuce_1 = pd.read_csv('../data/tijiao_part1.csv', header=None)
    yuce_1.columns = ['product_id','product_month','ciiquantity_month']
    yuce_3476 = yuce_1['product_id'].unique()
    yuce_4000 = range(1,4001)
    chaji = list(set(yuce_4000).difference(set(yuce_3476))) # a中有而b中没有的

    # 把之前的规则那部分也加上，看看能从填充0的187提高到多少
    guize = pd.read_csv('../data/new_2.txt')

    for i in chaji:
        data = guize[guize['product_id'] == i]
        data.to_csv('../data/tijiao_part2.csv', index=False, mode='a', encoding='utf-8', header=None)


# 这个函数用“随机森林”模型把结果做出来
def random_forest():
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')
    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']})

    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(n_estimators=500,max_depth=6) #167.829683

    '''都用'''
    col = list(test.columns[1:])

    for i in range(0,12):

        print('begin to pred which label ? ' , 'label_' + str(i))
        # 特征列
        one_line_train = col
        one_line_test = col
        # label列
        lie = 'num_' + str(i + 12)
        # 开gbdt模型
        feature_train = train[one_line_train] # 训练集的“特征集合”
        bizhi_lie = train[lie] # 训练集的“label列”

        print('begin to build model')
        clf.fit(feature_train.as_matrix(), bizhi_lie.as_matrix())  # 训练出来模型
        print clf.oob_prediction_
        print('build model over')

        prediction = clf.predict(test[one_line_test].as_matrix())
        prediction = DataFrame(prediction)
        prediction = prediction[prediction.columns[0]]

        Submission14[lie] = prediction.astype('int')

    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])

    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv' , index = False , encoding="utf-8" , mode='a')


# 这个函数用“随机森林”模型把结果做出来
def random_forest_kf():
    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 使用所有特征 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
    train = pd.read_csv('../data/train_fea_4.csv')
    test = pd.read_csv('../data/test_fea_4.csv')
    Submission14 = DataFrame({'product_id':test.loc[:,'product_id']})

    from sklearn.ensemble import RandomForestRegressor


    '''都用'''
    fea_all = list(test.columns[1:])


    for i in range(0,12):
        print('begin to pred which label ? ' , 'label_' + str(i))

        train_data = train[fea_all]
        train_data = np.array(train_data)  # 训练集（特征列）
        train_target = train['num_' + str(i + 12)]
        train_target = np.array(train_target)  # 训练集（label列）

        num_fold = 0
        random_state = 51
        models = []
        nfolds = 10
        kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)

        for train_index, test_index in kf:  # train_index test_index 在 5次循环中每一次都不同，但是每一次的这两个list合起来刚好是 0~len(train_data)中的每一个数字
            print(u'which model begin to make : ', num_fold)
            num_fold += 1
            clf = RandomForestRegressor(n_estimators=500, max_depth=6)  # 167.829683
            # n_estimators=1)
            X_train = train_data[train_index]
            X_train = DataFrame(X_train)  # 训练集的“特征列”（DataFrame类型）
            Y_train = train_target[train_index]
            Y_train = DataFrame(Y_train)
            Y_train = Y_train[0]  # 训练集的“label列”（Series类型）

            X_valid = train_data[test_index]
            X_valid = DataFrame(X_valid)  # 测试集的“特征列”（DataFrame类型）
            Y_valid = train_target[test_index]
            Y_valid = DataFrame(Y_valid)
            Y_valid = Y_valid[0]  # 测试集的“label列”（DataFrame类型）

            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))
            print('begin to make model')
            clf.fit(X_train.as_matrix(), Y_train.as_matrix())  # 训练出来模型
            print('make model over')

            models.append(clf)  # 将上面刚刚训练好的模型存储起来

        # 预测
        num_fold = 0
        yfull_test = []
        for j in range(nfolds):
            print(u'which model we will use to predicte : ', num_fold)
            model = models[j]  # 先拿第一个模型来搞
            num_fold += 1
            y_pred = model.predict(test[fea_all].as_matrix())

            yfull_test.append(list(y_pred))

        a = np.array(yfull_test[0])  # 第一个模型的预测结果
        for j in range(1, nfolds):
            a += np.array(yfull_test[j])
        a /= nfolds


        Submission14['num_' + str(i + 12)] = a


    for i in list(Submission14.columns)[1:]:
        Submission14[i] = abs(Submission14[i])

    Submission14['num_24'] = Submission14['num_12']
    Submission14['num_25'] = Submission14['num_13']
    Submission14.to_csv('../data/yuce.csv' , index = False , encoding="utf-8" , mode='a')

# 现在单模型能做到xgb 170.344683    random->了，规则还是173.486408，想几个加权融合下
def ronghe():
    model_xgb = pd.read_csv('../data/result/xgb_kf_162.csv')
    model_random = pd.read_csv('../data/result/random_kf.csv')
    guize = pd.read_csv('../data/new_2.txt')

    # 直接在 guize 这个文件中 进行修改数据
    for i in guize['product_id'].unique():
        # print(i)
        # 找出来 在 model 中这个产品的数据
        data_model_xgb = model_xgb[model_xgb['product_id'] == i]
        data_model_random = model_random[model_random['product_id'] == i]

        # 还要找出来 在 guize 中这个产品的数据，之所以要找出来这个数据是因为：一会进行加权的时候，有一个要求是两个进行加权的数据的 Index 一样
        data_guize = guize[guize['product_id'] == i].copy()
        # 修改data_model的索引
        data_model_xgb.index = data_guize.index
        data_model_random.index = data_guize.index


        # 进行修改
        guize.loc[guize['product_id'] == i, 'ciiquantity_month'] = data_guize['ciiquantity_month'] * 0.3+data_model_random['ciiquantity_month'] * 0.5+ \
                                                                   data_model_xgb['ciiquantity_month'] * 0.2

    guize.to_csv('../data/result/result_3_5_2.txt' , index=False , mode='a' , encoding='utf-8')
# 	result_3_6_1.txt	2017-04-15 08:46:58	  164.239616 可以试试



def yi_fangcha():
    import math
    model_xgb = pd.read_csv('../data/result/tijiao_xgb.csv')
    model_xgb['ciiquantity_month'] = model_xgb['ciiquantity_month'].apply(lambda x: math.exp(math.log(1*x+160)))
    print model_xgb.head()

    model_xgb.to_csv('../data/result/duibi_3.csv' , index=False , mode='a' , encoding='utf-8')

if __name__ == '__main__':


    random_forest_kf()
    xgb_kf()

    yuce_2_tijiao()

    try_2()
    try_1_1()



    ronghe()




