# coding=utf-8
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import pickle
import cPickle
from math import ceil
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

# 在这个文件中，我们首先用 backfill 的方式将那些不足23条数据的 产品填充到这么多，然后对总体数据进行建模
def add_to_23(data):
    for i in data['product_id'].unique():
        data_one = data[data['product_id'] == i]
        # 填充到23条
        a = range(1,24)
        a = DataFrame(a)
        a.columns = ['month']
        # 进行merge
        data_merge = pd.merge(a , data_one , on='month' , how='left')
        data_merge['product_id'] = data_merge['product_id'].fillna(method='ffill') # product_id列的fill顺序可以乱
        data_merge['product_id'] = data_merge['product_id'].fillna(method='bfill')

        data_merge['num'] = data_merge['num'].fillna(method='bfill') # 这个必须是先backfill先
        data_merge['num'] = data_merge['num'].fillna(method='ffill')
        data_merge = data_merge[['product_id','month','num']]

        if(i == 1):
            data_merge.to_csv('../data/month_fill_2_23.csv', index=False, mode='a', encoding='utf-8')
        else:
            data_merge.to_csv('../data/month_fill_2_23.csv', index=False, mode='a', encoding='utf-8',header=None) # 3476个产品


# 传进来的参数是“没有经过转置的训练集”
# 输出到 ../data/month_fill_2_23_T.csv
def make_T(month_fill_2_23):
    col = ['product_id']
    for i in range(1,24):
        col.append('num_' + str(i))
    for i in month_fill_2_23['product_id'].unique():
        # print(i)
        data = month_fill_2_23[month_fill_2_23['product_id'] == i]
        if(len(data) != 23): # 防错代码
            print i
        if(i == 1):
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['num'])
            list_25 = DataFrame(list_25).T
            list_25.columns = col
            list_25.to_csv('../data/month_fill_2_23_T.csv' , index=False , mode='a' , encoding='utf-8')
        else:
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['num'])
            list_25 = DataFrame(list_25).T

            list_25.to_csv('../data/month_fill_2_23_T.csv', index=False, mode='a', encoding='utf-8', header=None)

# 做出来预测集的基本数据
def make_test(month_fill_2_23_T):
    # 训练集值需要找出来前11列num值就可以了
    col = ['product_id']
    for i in range(13,24):
        col.append('num_'+str(i))
    data = month_fill_2_23_T[col]
    col_reset = ['product_id']
    for i in range(1,12):
        col_reset.append('num_'+str(i))
    data.columns = col_reset
    data.to_csv('../data/test.csv' , index=False , mode='a' , encoding='utf-8')

# 这个函数可以满足 训练集 都做出来第一部分特征
def make_train_fea_1(train):
    '''做出来几个均值'''
    train['ave_5'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11']) / 5

    train['ave_7'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11'] + train['num_6'] + train['num_5']) / 7

    train['ave_9'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11'] + train['num_6'] + train['num_5']+ train['num_4'] + \
                      train['num_3']) / 9

    train['ave_11'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                       train['num_11'] + train['num_6'] + train['num_5']+ train['num_4'] + \
                       train['num_3']+ train['num_2'] + train['num_1']) / 11

    '''将用户的基本属性添加上'''
    product_info = pd.read_csv('../data/product_info.csv')
    # 首先删掉一些列
    list_will_delete = ['railway','airport','citycenter','railway2','airport2','citycenter2','startdate','upgradedate',\
                        'cooperatedate']
    for i in list_will_delete:
        del product_info[i]
    # 填充一些列，填充的方法都是用众数来进行填充
    list_will_fill = ['district_id2','district_id2','lat','lon','eval2','eval3','eval4','voters','maxstock']
    for i in list_will_fill:
        product_info[i] = product_info[i].replace(-1,product_info[i].mode()[0])

    '''将两个文件 merge'''

    new = pd.merge(train , product_info , on='product_id' , how='left')
    new.to_csv('../data/train_fea_1.csv', index=False, mode='a', encoding='utf-8')

# 这个函数可以满足 预测集 都做出来第一部分特征
def make_test_fea_1(train):
    '''做出来几个均值'''
    train['ave_5'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11']) / 5

    train['ave_7'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11'] + train['num_6'] + train['num_5']) / 7

    train['ave_9'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                      train['num_11'] + train['num_6'] + train['num_5']+ train['num_4'] + \
                      train['num_3']) / 9

    train['ave_11'] = (train['num_7'] + train['num_8'] + train['num_9'] + train['num_10'] + \
                       train['num_11'] + train['num_6'] + train['num_5']+ train['num_4'] + \
                       train['num_3']+ train['num_2'] + train['num_1']) / 11

    '''将用户的基本属性添加上'''
    product_info = pd.read_csv('../data/product_info.csv')
    # 首先删掉一些列
    list_will_delete = ['railway','airport','citycenter','railway2','airport2','citycenter2','startdate','upgradedate',\
                        'cooperatedate']
    for i in list_will_delete:
        del product_info[i]
    # 填充一些列，填充的方法都是用众数来进行填充
    list_will_fill = ['district_id2','district_id2','lat','lon','eval2','eval3','eval4','voters','maxstock']
    for i in list_will_fill:
        product_info[i] = product_info[i].replace(-1,product_info[i].mode()[0])

    '''将两个文件 merge'''

    new = pd.merge(train , product_info , on='product_id' , how='left')
    new.to_csv('../data/test_fea_1.csv', index=False, mode='a', encoding='utf-8')



# 接下来就是将 ../data/month_order_num_2.csv 这个文件，和 ../data/month_fill_2_23.csv 两个进行下对比
# 然后按照 month_fill_2_23 的格式，将month_order_num_2这个文件处理下
def deal_order_num():
    month_order_num_2 = pd.read_csv('../data/month_order_num_2.csv',header=None)
    month_order_num_2.columns = ['product_id','order_num','month']

    month_num_2 = pd.read_csv('../data/month_fill_2_23.csv')

    for i in month_order_num_2['product_id'].unique():
        # print(i)
        data_1 = month_order_num_2[month_order_num_2['product_id'] == i]
        data_2 = month_num_2[month_num_2['product_id'] == i]
        data_2 = data_2[['month']]
        # 因为data_1 data_2两个数据集之间，有可能行数不一样，肯定要以month_num_2的为准，因为要一会添加特征
        new = pd.merge(data_2 , data_1 , on='month' , how='left')
        # 有可能有缺失数据
        new['product_id'] = new['product_id'].fillna(method='ffill')
        new['product_id'] = new['product_id'].fillna(method='bfill')
        new['order_num'] = new['order_num'].fillna(0)
        new = new[['product_id','month','order_num']]
        if(i == 1):
            new.to_csv('../data/month_order_num_2_new.csv', index=False , mode='a' , encoding='utf-8')
        else:
            new.to_csv('../data/month_order_num_2_new.csv', index=False, mode='a', encoding='utf-8',header=None)

# 对 month_order_num_2_new 做转置
def deal_order_num_T():
    # 转置
    month_order_num_2_new = pd.read_csv('../data/month_order_num_2_new.csv')
    col = ['product_id']
    for i in range(1, 24):
        col.append('order_num_' + str(i))
    for i in month_order_num_2_new['product_id'].unique():
        # print(i)
        data = month_order_num_2_new[month_order_num_2_new['product_id'] == i]
        if (len(data) != 23):
            print i, len(data)
        if (i == 1):
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['order_num'])
            list_25 = DataFrame(list_25).T
            list_25.columns = col
            list_25.to_csv('../data/month_order_num_2_new_T.csv', index=False, mode='a', encoding='utf-8')
        else:
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['order_num'])
            list_25 = DataFrame(list_25).T

            list_25.to_csv('../data/month_order_num_2_new_T.csv', index=False, mode='a', encoding='utf-8', header=None)


# 这个函数可以满足 训练集 做出来第二部分特征
# 参数 train_fea_1 是 ../data/train_fea_1.csv（第一批含有特征的训练集）
# month_order_num_2_new_T 是 ../data/month_order_num_2_new_T.csv（ 每月订购销售量 ）
def make_train_fea_2(train_fea_1 , month_order_num_2_new_T):

    '''加载 ../data/month_order_num_2_new_T.csv  取前11列 随后做出来4个均值'''
    train_data_1 = month_order_num_2_new_T
    col = ['product_id']
    for i in range(1,12):
        col.append('order_num_' + str(i))
    train_data_1 = train_data_1[col]

    train_data_1['order_ave_5'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11']) / 5

    train_data_1['order_ave_7'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']) / 7

    train_data_1['order_ave_9'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']+ train_data_1['order_num_4'] + \
                             train_data_1['order_num_3']) / 9

    train_data_1['order_ave_11'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']+ train_data_1['order_num_4'] + \
                             train_data_1['order_num_3']+ train_data_1['order_num_2'] + train_data_1['order_num_1']) / 11

    new = pd.merge(train_fea_1 , train_data_1 , on='product_id' , how='inner')
    new.to_csv('../data/train_fea_2.csv', index=False, mode='a', encoding='utf-8')



# 这个函数可以满足 预测集 做出来第二部分特征
# 参数 train_fea_1 是 ../data/test_fea_1.csv（第一批含有特征的训练集）
# month_order_num_2_new_T 是 ../data/month_order_num_2_new_T.csv（ 每月订购销售量 ）
def make_test_fea_2(train_fea_1 , month_order_num_2_new_T):
    '''加载 ../data/month_order_num_2_new_T.csv  取前11列 随后做出来4个均值'''
    train_data_1 = month_order_num_2_new_T
    col = ['product_id']
    for i in range(13,24):
        col.append('order_num_' + str(i))
    train_data_1 = train_data_1[col]

    col_reset = ['product_id']
    for i in range(1,12):
        col_reset.append('order_num_' + str(i))
    train_data_1.columns = col_reset

    train_data_1['order_ave_5'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11']) / 5

    train_data_1['order_ave_7'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']) / 7

    train_data_1['order_ave_9'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']+ train_data_1['order_num_4'] + \
                             train_data_1['order_num_3']) / 9

    train_data_1['order_ave_11'] = (train_data_1['order_num_7'] + train_data_1['order_num_8'] + train_data_1['order_num_9'] + train_data_1['order_num_10'] + \
                            train_data_1['order_num_11'] + train_data_1['order_num_6'] + train_data_1['order_num_5']+ train_data_1['order_num_4'] + \
                             train_data_1['order_num_3']+ train_data_1['order_num_2'] + train_data_1['order_num_1']) / 11

    new = pd.merge(train_fea_1 , train_data_1 , on='product_id' , how='inner')
    new.to_csv('../data/test_fea_2.csv', index=False, mode='a', encoding='utf-8')


# 防止第三部分特征中的数据不足，将其转换成和 month_fill_2_23 相同行的数据
def part_3_deal_1():

    month_order_num_2 = pd.read_csv('../data/price_2.csv',header=None)
    month_order_num_2.columns = ['product_id','month_price_zong','month']

    month_num_2 = pd.read_csv('../data/month_fill_2_23.csv')

    for i in month_order_num_2['product_id'].unique():
        # print(i)
        data_1 = month_order_num_2[month_order_num_2['product_id'] == i]
        data_2 = month_num_2[month_num_2['product_id'] == i]
        data_2 = data_2[['month']]
        # 因为data_1 data_2两个数据集之间，有可能行数不一样，肯定要以month_num_2的为准，因为要一会添加特征
        new = pd.merge(data_2 , data_1 , on='month' , how='left')
        # 有可能有缺失数据
        new['product_id'] = new['product_id'].fillna(method='ffill')
        new['product_id'] = new['product_id'].fillna(method='bfill')
        new['month_price_zong'] = new['month_price_zong'].fillna(0)
        new = new[['product_id','month','month_price_zong']]
        if(i == 1):
            new.to_csv('../data/price_3.csv', index=False , mode='a' , encoding='utf-8')
        else:
            new.to_csv('../data/price_3.csv', index=False, mode='a', encoding='utf-8',header=None)


# 对 price_3 做转置
def part_3_deal_1_T():
    month_order_num_2_new = pd.read_csv('../data/price_3.csv')
    col = ['product_id']
    for i in range(1,24):
        col.append('month_price_zong_' + str(i))
    for i in month_order_num_2_new['product_id'].unique():
        # print(i)
        data = month_order_num_2_new[month_order_num_2_new['product_id'] == i]
        if(len(data) != 23):
            print i,len(data)
        if(i == 1):
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['month_price_zong'])
            list_25 = DataFrame(list_25).T
            list_25.columns = col
            list_25.to_csv('../data/price_3_T.csv' , index=False , mode='a' , encoding='utf-8')
        else:
            data = data.sort_values(by=['month'])
            # 进行转置
            list_25 = list(data['product_id'].unique())
            list_25 = list_25 + list(data['month_price_zong'])
            list_25 = DataFrame(list_25).T

            list_25.to_csv('../data/price_3_T.csv', index=False, mode='a', encoding='utf-8', header=None)

# 训练集第三部分特征
# train_fea_2 ：第二部分特征
# price_3_T ：要添加到第二部分的特征
def make_train_fea_3(train_fea_2 , price_3_T):
    '''做出来几个均值'''
    train_data_1 = price_3_T
    col = ['product_id']
    for i in range(1,12):
        col.append('month_price_zong_'+str(i))
    train_data_1 = train_data_1[col]
    train_data_1['month_price_zong_ave_5'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11']) / 5

    train_data_1['month_price_zong_ave_7'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']) / 7

    train_data_1['month_price_zong_ave_9'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']+ train_data_1['month_price_zong_4'] + \
                             train_data_1['month_price_zong_3']) / 9

    train_data_1['month_price_zong_ave_11'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']+ train_data_1['month_price_zong_4'] + \
                             train_data_1['month_price_zong_3']+ train_data_1['month_price_zong_2'] + train_data_1['month_price_zong_1']) / 11

    '''将两个文件 merge'''
    new = pd.merge(train_fea_2 , train_data_1 , on='product_id' , how='inner')

    '''填充好'''
    train = new

    for i in range(1,12):
        train['num_' + str(i)] = train['num_' + str(i)].replace(0,1)

    # 可以提取特征了
    for i in range(1,12):
        train['div_' + str(i)] = train['month_price_zong_' + str(i)] / train['num_' + str(i)]

    train.to_csv('../data/train_fea_3.csv', index=False, mode='a', encoding='utf-8')


# 预测集第三部分特征
# train_fea_2 ：第二部分特征
# price_3_T ：要添加到第二部分的特征
def make_test_fea_3(train_fea_2 , price_3_T):
    '''做出来几个均值'''
    train_data_1 = price_3_T
    col = ['product_id']
    for i in range(13, 24):
        col.append('month_price_zong_' + str(i))
    train_data_1 = train_data_1[col]

    col_reset = ['product_id']
    for i in range(1, 12):
        col_reset.append('month_price_zong_' + str(i))
    train_data_1.columns = col_reset
    train_data_1['month_price_zong_ave_5'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11']) / 5

    train_data_1['month_price_zong_ave_7'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']) / 7

    train_data_1['month_price_zong_ave_9'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']+ train_data_1['month_price_zong_4'] + \
                             train_data_1['month_price_zong_3']) / 9

    train_data_1['month_price_zong_ave_11'] = (train_data_1['month_price_zong_7'] + train_data_1['month_price_zong_8'] + train_data_1['month_price_zong_9'] + train_data_1['month_price_zong_10'] + \
                            train_data_1['month_price_zong_11'] + train_data_1['month_price_zong_6'] + train_data_1['month_price_zong_5']+ train_data_1['month_price_zong_4'] + \
                             train_data_1['month_price_zong_3']+ train_data_1['month_price_zong_2'] + train_data_1['month_price_zong_1']) / 11

    '''将两个文件 merge'''
    new = pd.merge(train_fea_2 , train_data_1 , on='product_id' , how='inner')

    '''填充好'''
    train = new

    for i in range(1,12):
        train['num_' + str(i)] = train['num_' + str(i)].replace(0,1)

    # 可以提取特征了
    for i in range(1,12):
        train['div_' + str(i)] = train['month_price_zong_' + str(i)] / train['num_' + str(i)]

    train.to_csv('../data/test_fea_3.csv', index=False, mode='a', encoding='utf-8')


# 训练集“添加几个 评分差值 特征 eval,eval2,eval3,eval4 这几个特征使得分数下降了，不如不加”
def add_ping_fea():
    train_fea_3 = pd.read_csv('../data/train_fea_3.csv')
    train_fea_3['ping_1'] = train_fea_3['eval'] - train_fea_3['eval2']
    train_fea_3['ping_2'] = train_fea_3['eval'] - train_fea_3['eval3']
    train_fea_3['ping_3'] = train_fea_3['eval'] - train_fea_3['eval4']
    train_fea_3['ping_4'] = train_fea_3['eval2'] - train_fea_3['eval3']
    train_fea_3['ping_5'] = train_fea_3['eval2'] - train_fea_3['eval4']
    train_fea_3['ping_6'] = train_fea_3['eval3'] - train_fea_3['eval4']
    train_fea_3.to_csv('../data/train_fea_4.csv', index=False, mode='a', encoding='utf-8')
    train_fea_3 = pd.read_csv('../data/test_fea_3.csv')
    train_fea_3['ping_1'] = train_fea_3['eval'] - train_fea_3['eval2']
    train_fea_3['ping_2'] = train_fea_3['eval'] - train_fea_3['eval3']
    train_fea_3['ping_3'] = train_fea_3['eval'] - train_fea_3['eval4']
    train_fea_3['ping_4'] = train_fea_3['eval2'] - train_fea_3['eval3']
    train_fea_3['ping_5'] = train_fea_3['eval2'] - train_fea_3['eval4']
    train_fea_3['ping_6'] = train_fea_3['eval3'] - train_fea_3['eval4']
    train_fea_3.to_csv('../data/test_fea_4.csv', index=False, mode='a', encoding='utf-8')

# 计算这几列之间的差值
# num_1,num_2,num_3,num_4,num_5,num_6,num_7,num_8,num_9,num_10,num_11
# order_num_1,order_num_2,order_num_3,order_num_4,order_num_5,order_num_6,order_num_7,order_num_8,order_num_9,order_num_10,order_num_11
def month_cha():
    train_fea_3 = pd.read_csv('../data/train_fea_3.csv')
    for i in range(2,12):
        train_fea_3['cha_num_'+str(i)] = train_fea_3['num_'+str(i)] - train_fea_3['num_'+str(i-1)]
        # train_fea_3['cha_order_num_' + str(i)] = train_fea_3['order_num_' + str(i)] - train_fea_3['order_num_' + str(i - 1)]
    train_fea_3.to_csv('../data/train_fea_4.csv', index=False, mode='a', encoding='utf-8')

    train_fea_3 = pd.read_csv('../data/test_fea_3.csv')
    for i in range(2,12):
        train_fea_3['cha_num_' + str(i)] = train_fea_3['num_' + str(i)] - train_fea_3['num_' + str(i - 1)]
        # train_fea_3['cha_order_num_' + str(i)] = train_fea_3['order_num_' + str(i)] - train_fea_3[
        #     'order_num_' + str(i - 1)]
    train_fea_3.to_csv('../data/test_fea_4.csv', index=False, mode='a', encoding='utf-8')


# 将几个时间点加进去，时间点不是以直接的方式，在训练集中是和 2014-01-01的月份差，在预测集中是和 2015-01-01的月份差
def add_time_three():
    train_fea_3 = pd.read_csv('../data/train_fea_3.csv')
    product_info = pd.read_csv('../data/product_info.csv')
    product_info.loc[product_info['startdate'] != '-1','startdate_change'] = product_info.loc[product_info['startdate'] != '-1']['startdate'].\
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2014-01-01', "%Y-%m-%d") ).days/30.0)
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_change'] = product_info.loc[product_info['upgradedate'] != '-1']['upgradedate'].\
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2014-01-01', "%Y-%m-%d")).days / 30.0)
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_change'] = product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate'].\
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2014-01-01', "%Y-%m-%d")).days / 30.0)
    product_info['startdate_change'] = product_info['startdate_change'].fillna('-1')
    product_info['upgradedate_change'] = product_info['upgradedate_change'].fillna('-1')
    product_info['cooperatedate_change'] = product_info['cooperatedate_change'].fillna('-1')
    train_fea_3 = pd.merge(train_fea_3, product_info[['product_id','startdate_change','upgradedate_change','cooperatedate_change']], on='product_id', how='left')
    train_fea_3.to_csv('../data/train_fea_4.csv', index=False, mode='a', encoding='utf-8')

    train_fea_3 = pd.read_csv('../data/test_fea_3.csv')
    product_info = pd.read_csv('../data/product_info.csv')
    product_info.loc[product_info['startdate'] != '-1', 'startdate_change'] = \
    product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2015-01-01', "%Y-%m-%d")).days / 30.0)
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_change'] = \
    product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2015-01-01', "%Y-%m-%d")).days / 30.0)
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_change'] = \
    product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime('2015-01-01', "%Y-%m-%d")).days / 30.0)
    product_info['startdate_change'] = product_info['startdate_change'].fillna('-1')
    product_info['upgradedate_change'] = product_info['upgradedate_change'].fillna('-1')
    product_info['cooperatedate_change'] = product_info['cooperatedate_change'].fillna('-1')
    train_fea_3 = pd.merge(train_fea_3, product_info[
        ['product_id', 'startdate_change', 'upgradedate_change', 'cooperatedate_change']], on='product_id', how='left')
    train_fea_3.to_csv('../data/test_fea_4.csv', index=False, mode='a', encoding='utf-8')


def add_time_9():
    import time
    train_fea_3 = pd.read_csv('../data/train_fea_3.csv')
    product_info = pd.read_csv('../data/product_info.csv')
    product_info.loc[product_info['startdate'] != '-1','startdate_year'] = product_info.loc[product_info['startdate'] != '-1']['startdate'].\
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['startdate'] != '-1', 'startdate_month'] = product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['startdate'] != '-1', 'startdate_day'] = product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))

    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_year'] = product_info.loc[product_info['upgradedate'] != '-1']['upgradedate'].\
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_month'] = product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_day'] = product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))

    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_year'] = product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate'].\
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_month'] = product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_day'] = product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))


    product_info['startdate_year'] = product_info['startdate_year'].fillna('-1')
    product_info['startdate_month'] = product_info['startdate_month'].fillna('-1')
    product_info['startdate_day'] = product_info['startdate_day'].fillna('-1')
    product_info['upgradedate_year'] = product_info['upgradedate_year'].fillna('-1')
    product_info['upgradedate_month'] = product_info['upgradedate_month'].fillna('-1')
    product_info['upgradedate_day'] = product_info['upgradedate_day'].fillna('-1')
    product_info['cooperatedate_year'] = product_info['cooperatedate_year'].fillna('-1')
    product_info['cooperatedate_month'] = product_info['cooperatedate_month'].fillna('-1')
    product_info['cooperatedate_day'] = product_info['cooperatedate_day'].fillna('-1')
    list_now = ['product_id','startdate_year','startdate_month','startdate_day','upgradedate_year','upgradedate_month','upgradedate_day', \
            'cooperatedate_year','cooperatedate_month','cooperatedate_day']
    train_fea_3 = pd.merge(train_fea_3, product_info[list_now], on='product_id', how='left')
    train_fea_3.to_csv('../data/train_fea_4.csv', index=False, mode='a', encoding='utf-8')

    train_fea_3 = pd.read_csv('../data/test_fea_3.csv')
    product_info = pd.read_csv('../data/product_info.csv')
    product_info.loc[product_info['startdate'] != '-1', 'startdate_year'] = \
    product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['startdate'] != '-1', 'startdate_month'] = \
    product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['startdate'] != '-1', 'startdate_day'] = \
    product_info.loc[product_info['startdate'] != '-1']['startdate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))

    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_year'] = \
    product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_month'] = \
    product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['upgradedate'] != '-1', 'upgradedate_day'] = \
    product_info.loc[product_info['upgradedate'] != '-1']['upgradedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))

    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_year'] = \
    product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_year))
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_month'] = \
    product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mon))
    product_info.loc[product_info['cooperatedate'] != '-1', 'cooperatedate_day'] = \
    product_info.loc[product_info['cooperatedate'] != '-1']['cooperatedate']. \
        apply(lambda a: (time.strptime(a, "%Y-%m-%d").tm_mday))

    product_info['startdate_year'] = product_info['startdate_year'].fillna('-1')
    product_info['startdate_month'] = product_info['startdate_month'].fillna('-1')
    product_info['startdate_day'] = product_info['startdate_day'].fillna('-1')
    product_info['upgradedate_year'] = product_info['upgradedate_year'].fillna('-1')
    product_info['upgradedate_month'] = product_info['upgradedate_month'].fillna('-1')
    product_info['upgradedate_day'] = product_info['upgradedate_day'].fillna('-1')
    product_info['cooperatedate_year'] = product_info['cooperatedate_year'].fillna('-1')
    product_info['cooperatedate_month'] = product_info['cooperatedate_month'].fillna('-1')
    product_info['cooperatedate_day'] = product_info['cooperatedate_day'].fillna('-1')
    list_now = ['product_id', 'startdate_year', 'startdate_month', 'startdate_day', 'upgradedate_year',
                'upgradedate_month', 'upgradedate_day', \
                'cooperatedate_year', 'cooperatedate_month', 'cooperatedate_day']
    train_fea_3 = pd.merge(train_fea_3, product_info[list_now], on='product_id', how='left')
    train_fea_3.to_csv('../data/test_fea_4.csv', index=False, mode='a', encoding='utf-8')

if __name__ == '__main__':

    month_num_2 = pd.read_csv('../data/month_num_2.csv', header=None)
    month_num_2.columns = ['product_id', 'num', 'month']
    add_to_23(month_num_2)

    month_fill_2_23_T = pd.read_csv('../data/month_fill_2_23.csv')
    make_T(month_fill_2_23_T)

    month_fill_2_23_T = pd.read_csv('../data/month_fill_2_23_T.csv')
    make_train(month_fill_2_23_T)
    make_test(month_fill_2_23_T)

    train = pd.read_csv('../data/month_fill_2_23_T.csv')
    make_train_fea_1(train)
    test = pd.read_csv('../data/test.csv')
    make_test_fea_1(test)

    deal_order_num()
    deal_order_num_T()


    month_order_num_2_new_T = pd.read_csv('../data/month_order_num_2_new_T.csv')
    train_fea_1 = pd.read_csv('../data/train_fea_1.csv')
    test_fea_1 = pd.read_csv('../data/test_fea_1.csv')
    make_train_fea_2(train_fea_1 , month_order_num_2_new_T)
    make_test_fea_2(test_fea_1 , month_order_num_2_new_T)

    part_3_deal_1()
    part_3_deal_1_T()

    price_3_T = pd.read_csv('../data/price_3_T.csv')
    train_fea_2 = pd.read_csv('../data/train_fea_2.csv')
    test_fea_2 = pd.read_csv('../data/test_fea_2.csv')
    make_train_fea_3(train_fea_2 , price_3_T)
    make_test_fea_3(test_fea_2 , price_3_T)

    add_time_three()



