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

'''统计下每一个产品 每一个月的销售量，同时将那些每一个月销量都是0的删掉'''
product_quantity = pd.read_csv('../data/product_quantity.csv')
product_quantity = product_quantity.sort_values(by=['product_id','product_date'])
product_quantity = product_quantity.reset_index(drop=True)
# 只截取出来 前7个字符
product_quantity['product_date'] = product_quantity['product_date'].str[:7]
# 根据 product_id product_date 这两列，进行求和
linshi = product_quantity.groupby(['product_id','product_date']).sum()
# 做处理一个文件，只含有 product_id product_date 和 对应的 ciiquantity 求和
month_num = DataFrame()
month_num['num'] = linshi.reset_index(drop=True)['ciiquantity']

linshi_1 = DataFrame(linshi.index)
linshi_1[0] = linshi_1[0].astype('str')

month_num['product_id'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
month_num['month'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1].str.split('\'').str[1]

month_num['product_id'] = month_num['product_id'].astype('int')

# 调整列名
month_num = month_num[['product_id' , 'month' , 'num']]


'''第二部分,将上面的月份换成对应的数字，这样子在画图的时候方便些'''
num = 1
for i in month_num['month'].unique():
    print i
    month_num.loc[month_num['month'] == i , 'month'] = num
    num += 1

for i in month_num_1['product_id'].unique():
    print(i)
    # 一个产品的数据
    data = month_num_1[month_num_1['product_id'] == i]
    '''为了防止上面的那种情况'''
    data = data[data['num'] != 0]
    if(len(data) == 0): # 防止有些数据除去0后，一条都没了！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        continue
    # 这个产品的最大最小月份
    month_min = data['month'].min()
    month_max = data['month'].max()
    # 用range函数做出来，填充后的月份

    month_tian = range(month_min , month_max+1)
    month_tian = DataFrame(month_tian)
    month_tian.columns = ['month']

    # merge出来一个完整的数据集，确缺失值具体处理
    data_merge = pd.merge(month_tian , data , on='month' , how='left')
    data_merge['product_id'] = data_merge['product_id'].fillna(method='ffill')
    data_merge['product_id'] = data_merge['product_id'].fillna(method='bfill')
    data_merge['num'] = data_merge['num'].fillna(0)

    for i in data_merge.columns:
        data_merge[i] = data_merge[i].astype('int')
    data_merge = data_merge[['product_id','num','month']]

    data_merge.to_csv('../data/month_num_2.csv', index=False , mode='a' , encoding='utf-8' , header=None)



'''
首先总结下现在的思路，../data/month_num_2.csv 这个文件中的数据中, 每一个产品的几个月销量都是相连的(已经做了除0变化，就是把全部是0的删除掉了)
我现在的想法是：如果哪个产品的最大月份不低于20，我就当做这个产品的 21 22 23这3个月的销售量是缺失，就用最大的那个值代替
对已填充完后 产品中最大月份是23，对这类产品用统一的规则
'''
'''第一步'''
month_num_2 = pd.read_csv('../data/month_num_2.csv',header=None)
month_num_2.columns = ['product_id','num','month']
# 首先找到那些最大月份 >= 20的产品，输出到文件中 data_1.csv
for i in month_num_2['product_id'].unique():
    data = month_num_2[month_num_2['product_id'] == i]
    if(data['month'].max() >= 20):
        if(i == 1):
            data.to_csv('../data/data_1.csv', index=False, mode='a', encoding='utf-8')
        else:
            data.to_csv('../data/data_1.csv', index=False, mode='a', encoding='utf-8',header=None)



'''第二步'''
data_1 = pd.read_csv('../data/data_1.csv')
# 对每一个产品填充到23
for i in data_1['product_id'].unique():
    data = data_1[data_1['product_id'] == i]
    # 做出来最大的销售值
    num_max = data['num'].max()

    # 这个产品的最大最小月份
    month_min = data['month'].min()

    month_tian = range(month_min , 24)
    month_tian = DataFrame(month_tian)
    month_tian.columns = ['month']

    # merge出来一个完整的数据集，确缺失值具体处理
    data_merge = pd.merge(month_tian , data , on='month' , how='left')
    data_merge['product_id'] = data_merge['product_id'].fillna(method='ffill')
    data_merge['product_id'] = data_merge['product_id'].fillna(method='bfill')
    data_merge['num'] = data_merge['num'].fillna(num_max)

    for i in data_merge.columns:
        data_merge[i] = data_merge[i].astype('int')
    data_merge = data_merge[['product_id','num','month']]

    data_merge.to_csv('../data/data_2.csv', index=False, mode='a', encoding='utf-8', header=None)

# # 查看下有多少产品会用到规则
data_2 = pd.read_csv('../data/data_2.csv',header=None)
data_2.columns = ['product_id','num','month']
print(len(data_2['product_id'].unique())) # 3450



'''找出来每个产品中月份 >= 12的销售月，也就是排完序后取最后12条数据'''
data_2 = pd.read_csv('../data/data_2.csv', header=None)
data_2.columns = ['product_id','num','month']

for i in data_2['product_id'].unique():
    # print(i)
    data = data_2[data_2['product_id'] == i]
    # 做排序
    data = data.sort_values(by='month')
    data = data.tail(12)
    if(i == 1):
        data.to_csv('../data/data_3.csv', index=False, mode='a', encoding='utf-8')
    else:
        data.to_csv('../data/data_3.csv', index=False, mode='a', encoding='utf-8', header=None)


# 先做出来 产品表的“雏形”
data_3 = pd.read_csv('../data/data_3.csv')

for i in data_3['product_id'].unique():
    print(i)
    data = data_3[data_3['product_id'] == i]
    # 每一行对应的权重
    data['quanzhong'] = (data['month'] - 11) ** 0.3
    # 根据权重 * 销售量，为一会最终的平均值做出来“中间值”
    data['ave'] = data['num'] * data['quanzhong']
    # 最终均值
    data['ave'] = data['ave'].sum() / data['quanzhong'].sum()

    if(i == 1):
        data.to_csv('../data/data_3_5.csv', index=False, mode='a', encoding='utf-8')
    else:
        data.to_csv('../data/data_3_5.csv', index=False, mode='a', encoding='utf-8', header=None)


# 接着再来一部, 做出来“产品表”
data_3_5 = pd.read_csv('../data/data_3_5.csv')
data_3_5 = data_3_5.groupby(by='product_id').mean()[['ave']]
data_3_5['product_id'] = data_3_5.index
data_3_5.to_csv('../data/data_3_5_5.csv', index=False, mode='a', encoding='utf-8')



# 做出来“月份比值”表
month_num_2 = pd.read_csv('../data/month_num_2.csv',header=None)
month_num_2.columns = ['product_id','num_bi','month']
month_num_2 = month_num_2[month_num_2['month'] >= 12] # 找出来大于12的数据

month_num_2 = month_num_2.groupby(by=['month']).sum()[['num_bi']]

# 将索引当做“月份”， 添加一各月份列（1~12）
month_num_2['yuefen'] = month_num_2.index
month_num_2['yuefen'] = month_num_2['yuefen'] - 12
month_num_2.loc[month_num_2['yuefen'] == 0,'yuefen'] = 12
# 有个比值型的
month_num_2['num_bi'] = month_num_2['num_bi'] / month_num_2['num_bi'].mean()
month_num_2.to_csv('../data/data_3_5_6.csv', index=False, mode='a', encoding='utf-8')


'''建立预测表，开始先建立“product_id”列、“月份列”、“日期列”'''
# 这里面就是要做的“那些产品”
data_3 = pd.read_csv('../data/data_3.csv') # product_id,num,month
# 将一个完好的文件加载下
yangben = pd.read_csv('../data/yangben.txt')
# 挑选出来我们要提交的那些用户数据-》只选取 产品id、月份两列
# 先求差集
data_3_id = list(data_3['product_id'].unique())
yangben_id = list(yangben['product_id'].unique())
chaji = list(set(yangben_id).difference(set(data_3_id))) # b中有而a中没有的
# 排除掉那些本次不存在的用户
for i in chaji:
    yangben = yangben[yangben['product_id'] != i]
# 只要前2列
yangben = yangben[['product_id', 'product_month']]

# 新添加一列，作为月份
yangben['yuefen'] = yangben['product_month'].str[5:7]
yangben['yuefen'] = yangben['yuefen'].astype('int')

yangben.to_csv('../data/data_4.csv', index=False, mode='a', encoding='utf-8')


'''接下来就可以往预测表里面添加很多东西了'''
data_4 = pd.read_csv('../data/data_4.csv') # 預測表     product_id,product_month,yuefen
data_3 = pd.read_csv('../data/data_3.csv') # 產品月份表 product_id,num,month
data_3['yuefen'] = data_3['month'] - 12
data_3.loc[data_3['yuefen'] == 0 , 'yuefen'] = 12
del data_3['month']

data_merge = pd.merge(data_4, data_3, on=['product_id','yuefen'],how='left')
print(len(data_4))
print(len(data_merge))
data_merge.to_csv('../data/data_4_1.csv', index=False, mode='a', encoding='utf-8')


# 其中的 产品表 就是我的 data_3_5_5，不过加载后需要把 month 这一列处理成“yuefen”列的形式
data_4 = pd.read_csv('../data/data_4_1.csv') # 预测表  product_id,product_month,yuefen,num
data_3 = pd.read_csv('../data/data_3_5_5.csv') # 产品月份表 ave,product_id

data_merge = pd.merge(data_4, data_3, on=['product_id'],how='left')
print(len(data_4))
print(len(data_merge))
data_merge.to_csv('../data/data_4_2.csv', index=False, mode='a', encoding='utf-8')


# 其中的 产品表 就是我的 data_3_5_5，不过加载后需要把 monrh 这一列处理成“yuefen”列的形式
data_4 = pd.read_csv('../data/data_4_2.csv') # 预测表  product_id,product_month,yuefen,num,ave
data_3 = pd.read_csv('../data/data_3_5_6.csv') # 产品月份表 num_bi,yuefen

data_merge = pd.merge(data_4, data_3, on=['yuefen'],how='left')
print(len(data_4))
print(len(data_merge))
data_merge.to_csv('../data/data_4_3.csv', index=False, mode='a', encoding='utf-8')


'''尝试再添加一种均值，和现在的相比就只是“权重变了”'''
# 先做出来 产品表的“雏形”
data_3 = pd.read_csv('../data/data_3.csv')

for i in data_3['product_id'].unique():
    print(i)
    data = data_3[data_3['product_id'] == i]
    # 每一行对应的权重
    # data['quanzhong'] = (data['month'] - 11) / 78
    data['quanzhong'] = (data['month'] - 11) / 12
    # 根据权重 * 销售量，为一会最终的平均值做出来“中间值”
    data['ave'] = data['num'] * data['quanzhong']
    # 最终均值
    data['ave'] = data['ave'].sum() / data['quanzhong'].sum()

    if(i == 1):
        data.to_csv('../data/data_3_5_new.csv', index=False, mode='a', encoding='utf-8')
    else:
        data.to_csv('../data/data_3_5_new.csv', index=False, mode='a', encoding='utf-8', header=None)


# 接着再来一部, 做出来“产品表”
data_3_5 = pd.read_csv('../data/data_3_5_new.csv')
data_3_5 = data_3_5.groupby(by='product_id').mean()[['ave']]
data_3_5['product_id'] = data_3_5.index
data_3_5.to_csv('../data/data_3_5_5_new.csv', index=False, mode='a', encoding='utf-8')



'''将../data/ceshi/data_3_5_5.csv添加到 ../data/ceshi/data_4_3.csv 上'''
data_3_5_5 = pd.read_csv('../data/data_3_5_5_new.csv')
data_3_5_5.columns = ['ave_2','product_id']

predict = pd.read_csv('../data/data_4_3.csv') # product_id,product_month,yuefen,num,ave,num_bi

predict = pd.merge(predict, data_3_5_5, on='product_id', how='left')

predict['pred'] = predict['ave'] * 0.35 + 0.3 * predict['num'] + predict['ave_2'] * 0.35

predict['pred'] = predict['pred'].fillna(predict['ave'] * predict['num_bi'])

predict.to_csv('../data/data_5.csv', index=False, mode='a', encoding='utf-8')


'''将上面做的这部分产品的数据汇总下 - 3450个产品的'''
data_5 = pd.read_csv('../data/data_5.csv')
col = ['product_id','product_month','pred']
data_5 = data_5[col]
data_5.columns = ['product_id','product_month','ciiquantity_month']

id_1 = list(data_5['product_id'].unique())
id_2 = range(1,4001)
chaji = list(set(id_2).difference(set(id_1))) # b中有而a中没有的

# 这类产品全部置0
list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
print(len(chaji)) # 37
for i in chaji:
    new = DataFrame()
    new['product_month'] = list(list_month)
    new['product_id'] = i
    new['ciiquantity_month'] = 0
    new = new[['product_id', 'product_month', 'ciiquantity_month']]
    new.to_csv('../data/data_' + str(len(chaji)) + '.csv', index=False, mode='a', encoding='utf-8', header=None)

# # 将 data_5中的 ciiquantity_month 取整
# data_5['ciiquantity_month'] = data_5['ciiquantity_month'].astype('int')

data_5.to_csv('../data/data_' + str(len(id_1)) + '.csv', index=False, mode='a', encoding='utf-8')




















'''上面用详细的规则做出来了一大部分数据质量比较好的一些产品，还有其余的一些用简单规则来做下'''
'''做出来差的那几个产品（524个产品）'''
month_num_2 = pd.read_csv('../data/month_num_2.csv',header=None)
month_num_2.columns = ['product_id' , 'num' , 'month']
# 差集
chaji = list(set(  range(1,4001)  ).difference(set(  list(month_num_2['product_id'].unique())   ))) # b中有而a中没有的
chaji = DataFrame(chaji)
chaji.to_csv('../data/524.csv', index=False, mode='a', encoding='utf-8' , header=None)


'''++++++++++++++++++++++++找出来那524个产品的属性++++++++++++++++++++++++'''
# 加载那505个产品
info_524 = pd.read_csv('../data/part_3/524.csv' , header=None)
info_524.columns = ['product_id']
# 加载产品信息表
product_info = pd.read_csv('../data/product_info.csv')
for i in info_524['product_id'].unique():
    print(i)
    data = product_info[product_info['product_id'] == i]
    data.to_csv('../data/info_524.csv', index=False, mode='a', encoding='utf-8' , header=None)



'''首先找出来43（22+21）个开业时间为缺失的数据'''
list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
info_505 = pd.read_csv('../data/part_3/info_524.csv' , header=None)
# 做出来上面的header
product_info = pd.read_csv('../data/product_info.csv')
info_505.columns = product_info.columns
data_1 = info_505[info_505['startdate'] == '-1'] # 22
data_2 = info_505[info_505['startdate'] == '1753-01-01'] # 21
# 对于则部分开业时间是缺失的产品--依旧置0
data_1_2 = data_1.append(data_2)
chaji = list(data_1_2['product_id'])
for i in chaji:
    new = DataFrame()
    new['product_month'] = list(list_month)
    new['product_id'] = i
    new['ciiquantity_month'] = 0
    new = new[['product_id', 'product_month', 'ciiquantity_month']]
    new.to_csv('../data/queshi_data.csv', index=False, mode='a', encoding='utf-8', header=None)



'''接下来是第二部分--开业时间和 2015-12-01日相差较远的数据，这里我将这个 相差月份定义为5个月'''
# 先排除掉 -1 1753-01-01 的数据后，找到那些开业时间比2015-12-1日还早5个月之外的数据
info_505 = pd.read_csv('../data/info_524.csv' , header=None)
# 做出来上面的header
product_info = pd.read_csv('../product_info.csv')
info_505.columns = product_info.columns
# 排除缺失数据
info_505 = info_505[info_505['startdate'] != '-1'] # 22
info_505 = info_505[info_505['startdate'] != '1753-01-01'] # 21
# 找到那些开业时间比2015-12-1日还早5个月之外的数据
data_zao = info_505[info_505['startdate'] < '2015-07-01'] # 67
# 这类产品全部置0
list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
chaji = list(data_zao['product_id'])
for i in chaji:
    new = DataFrame()
    new['product_month'] = list(list_month)
    new['product_id'] = i
    new['ciiquantity_month'] = 0
    new = new[['product_id', 'product_month', 'ciiquantity_month']]
    new.to_csv('../data/zao_data.csv', index=False, mode='a', encoding='utf-8', header=None)




'''接下来就是剩余的用户了，应该是 524-43-67 = 414（个）'''
info_524 = pd.read_csv('../data/info_524.csv' , header=None)
# 做出来上面的header
product_info = pd.read_csv('../data/product_info.csv')
info_524.columns = product_info.columns
# 排除掉上面的2部分用户
queshi_data = pd.read_csv('../data/queshi_data.csv' , header=None)
queshi_data.columns = ['product_id','product_month','ciiquantity_month']
zao_data = pd.read_csv('../data/zao_data.csv' , header=None)
zao_data.columns = ['product_id','product_month','ciiquantity_month']

# >>> len(queshi_data['product_id'].unique())
# 43
# >>> len(zao_data['product_id'].unique())
# 67

queshi_data_product_id = list(queshi_data['product_id'].unique())
zao_data_product_id = list(zao_data['product_id'].unique())

for i in queshi_data_product_id:
    info_524 = info_524[info_524['product_id'] != i]
for i in zao_data_product_id:
    info_524 = info_524[info_524['product_id'] != i]

info_524.to_csv('../data/info_414.csv', index=False, mode='a', encoding='utf-8')




'''接下来就是在 含有月份数据超过 14个月的那些产品中，找到每一个产品最相近的几个产品，然后用他们的均值来填充 开业后的月份'''
'''首先找出来那 2448 个产品的产品属性的数据（info_2448.csv）'''
# 首先找到那 含有月份数据超过 14个月的所有产品
fifth_part_new_1 = pd.read_csv('../data/part_1/1/fifth_part_new_new.csv',header=None)
fifth_part_new_2 = pd.read_csv('../data/part_1/2/fifth_part_new_new.csv',header=None)
fifth_part_new_3 = pd.read_csv('../data/part_1/3/fifth_part_new_new.csv',header=None)
fifth_part_new_4 = pd.read_csv('../data/part_1/4/fifth_part_new_new.csv',header=None)
fifth_part_new_5 = pd.read_csv('../data/part_1/5/fifth_part_new_new.csv',header=None)
fifth_part_new_6 = pd.read_csv('../data/part_1/6/fifth_part_new_new.csv',header=None)
fifth_part_new_7 = pd.read_csv('../data/part_1/7/fifth_part_new_new.csv',header=None)
fifth_part_new_8 = pd.read_csv('../data/part_1/8/fifth_part_new_new.csv',header=None)
fifth_part_new_9 = pd.read_csv('../data/part_1/9/fifth_part_new_new.csv',header=None)
col = ['product_id','product_month','ciiquantity_month']
fifth_part_new_1.columns = col
fifth_part_new_2.columns = col
fifth_part_new_3.columns = col
fifth_part_new_4.columns = col
fifth_part_new_5.columns = col
fifth_part_new_6.columns = col
fifth_part_new_7.columns = col
fifth_part_new_8.columns = col
fifth_part_new_9.columns = col
# 单独做出来这批用户，然后将这批用户的数据存储为 info_.csv
fifth_part_new_1_id = list(fifth_part_new_1['product_id'].unique())
fifth_part_new_2_id = list(fifth_part_new_2['product_id'].unique())
fifth_part_new_3_id = list(fifth_part_new_3['product_id'].unique())
fifth_part_new_4_id = list(fifth_part_new_4['product_id'].unique())
fifth_part_new_5_id = list(fifth_part_new_5['product_id'].unique())
fifth_part_new_6_id = list(fifth_part_new_6['product_id'].unique())
fifth_part_new_7_id = list(fifth_part_new_7['product_id'].unique())
fifth_part_new_8_id = list(fifth_part_new_8['product_id'].unique())
fifth_part_new_9_id = list(fifth_part_new_9['product_id'].unique())
id_all = fifth_part_new_1_id + fifth_part_new_2_id
id_all = id_all + fifth_part_new_3_id
id_all = id_all + fifth_part_new_4_id
id_all = id_all + fifth_part_new_5_id
id_all = id_all + fifth_part_new_6_id
id_all = id_all + fifth_part_new_7_id
id_all = id_all + fifth_part_new_8_id
id_all = id_all + fifth_part_new_9_id # len(id_all) = 2448

product_info = pd.read_csv('../data/product_info.csv')
for i in id_all:
    data = product_info[product_info['product_id'] == i]
    if(i == 1):
        data.to_csv('../data/info_2448.csv', index=False, mode='a', encoding='utf-8')
    else:
        data.to_csv('../data/info_2448.csv', index=False, mode='a', encoding='utf-8' ,header=None )


'''把2448个产品的对应数据在 1~9 文件夹中对应找出来，为了方面把那9个文件已经放在了 ./fifth_part_new_new 文件夹下'''
'''对应的文件名是：fifth_part_1.csv ~ fifth_part_9.csv'''
'''再把第2部分的产品数据也做出来，然后将两个文件合二为一'''
# 但是由于这9个文件的列数不一样多（因为这是 月数据>14个月的产品，有的是15个月，有的是23个月）
# 所以我只挑出来他们都具备的 几列， product_id、 num_9 ~ num_23
col = ['product_id']
for i in range(12,24):
    col.append('num_' + str(i))
fifth_part_1 = pd.read_csv('../data/part_1/1/fifth_part.csv')
fifth_part_2 = pd.read_csv('../data/part_1/2/fifth_part.csv')
fifth_part_3 = pd.read_csv('../data/part_1/3/fifth_part.csv')
fifth_part_4 = pd.read_csv('../data/part_1/4/fifth_part.csv')
fifth_part_5 = pd.read_csv('../data/part_1/5/fifth_part.csv')
fifth_part_6 = pd.read_csv('../data/part_1/6/fifth_part.csv')
fifth_part_7 = pd.read_csv('../data/part_1/7/fifth_part.csv')
fifth_part_8 = pd.read_csv('../data/part_1/8/fifth_part.csv')
fifth_part_9 = pd.read_csv('../data/part_1/9/fifth_part.csv')
fifth_part_1 = fifth_part_1[col]
fifth_part_2 = fifth_part_2[col]
fifth_part_3 = fifth_part_3[col]
fifth_part_4 = fifth_part_4[col]
fifth_part_5 = fifth_part_5[col]
fifth_part_6 = fifth_part_6[col]
fifth_part_7 = fifth_part_7[col]
fifth_part_8 = fifth_part_8[col]
fifth_part_9 = fifth_part_9[col]
fifth_part = fifth_part_1.append(fifth_part_2)
fifth_part = fifth_part.append(fifth_part_3)
fifth_part = fifth_part.append(fifth_part_4)
fifth_part = fifth_part.append(fifth_part_5)
fifth_part = fifth_part.append(fifth_part_6)
fifth_part = fifth_part.append(fifth_part_7)
fifth_part = fifth_part.append(fifth_part_8)
fifth_part = fifth_part.append(fifth_part_9)
# 做出来几个均值
fifth_part['ave_12'] = (fifth_part['num_14'] + fifth_part['num_15'] + fifth_part['num_16'] + fifth_part['num_17'] + fifth_part['num_18'] + fifth_part['num_19'] + \
                               fifth_part['num_20'] + fifth_part['num_21'] + fifth_part['num_22'] + \
                             fifth_part['num_23'] + fifth_part['num_13'] + fifth_part['num_12']  ) / 12
fifth_part['ave_12'] = fifth_part['ave_12'].astype('int')
# 只挑选出来2列
fifth_part = fifth_part[['product_id' , 'ave_12']]
#
#
# 第二部分产品！！
#
chaji_data = pd.read_csv('../data/chaji_data.csv' , header=None)
chaji_data.columns = ['product_id' , 'num' , 'month']
chaji_data_sum = chaji_data.groupby(['product_id']).sum()
chaji_data_sum['product_id'] = chaji_data_sum.index
chaji_data_sum = chaji_data_sum.reset_index(drop=True)
chaji_data_sum = chaji_data_sum[['product_id' , 'num']]
# 上面的num列是总和，还应该除以“几个月”（但是首先要排除掉月份为0的数据，然后最大减最小）
for i in chaji_data_sum['product_id'].unique():
    data = chaji_data[chaji_data['product_id'] == i]
    divi = data['month'].max() - data['month'].min()
    divi += 1
    chaji_data_sum.loc[chaji_data_sum['product_id'] == i , 'num'] = chaji_data_sum.loc[chaji_data_sum['product_id'] == i , 'num'] / divi
chaji_data_sum.columns = ['product_id' , 'ave_12']
chaji_data_sum['ave_12'] = chaji_data_sum['ave_12'].astype('int')
# 合并！！！！！！！！！！！
fifth_part = fifth_part.append(chaji_data_sum) # 2448 + 1028 = 3476

fifth_part.to_csv('../data/data_2448_1028.csv', index=False, mode='a', encoding='utf-8' )


# 对应将上面的 用户的基本信息也输出到一个文件中 info_2448_1028.csv
data_2448_1028 = pd.read_csv('../data/data_2448_1028.csv')
product_info = pd.read_csv('../data/product_info.csv')

for i in data_2448_1028['product_id'].unique():
    data = product_info[product_info['product_id'] == i]
    if(i == 1):
        data.to_csv('../data/info_2448_1028.csv', index=False, mode='a', encoding='utf-8' )
    else:
        data.to_csv('../data/info_2448_1028.csv', index=False, mode='a', encoding='utf-8', header=None)


'''接下来就是在这 2448+1028 个产品中找到那 414(524个原始产品，减去43个开业时间为缺失的，再减去67个开业时间离得相差5个月的 = 414)
 个产品中每一个最相近的一批用户'''

list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']

data_2448_1028 = pd.read_csv('../data/data_2448_1028.csv')
# 选用的属性有：district_id1,district_id2,district_id3,district_id4,lat,lon,eval,eval2,eval3,eval4,voters,maxstock
# 先试下 完全相等来找用户，看看403个产品中每一个用户可以找到几个相近的用户！！！
info_414 = pd.read_csv('../data/info_414.csv')
info_2448_1028 = pd.read_csv('../data/info_2448_1028.csv')
for i in info_414['product_id'].unique():
    print(i)
    # 先找到414个产品中  当次循环的产品数据（仅仅是1行）
    dang_data = info_414[info_414['product_id'] == i]
    dang_data = dang_data.reset_index(drop=True)
    data = info_2448_1028[info_2448_1028['district_id1'] == dang_data['district_id1'][0]]
    data = data[data['district_id2'] == dang_data['district_id2'][0]]
    data = data[data['eval'] == dang_data['eval'][0]]
    data = data[data['eval2'] == dang_data['eval2'][0]]
    data = data[data['eval3'] <= (dang_data['eval3'][0] + 1)]
    data = data[data['eval3'] >= (dang_data['eval3'][0] - 1)]
    data = data[data['eval4'] <= (dang_data['eval4'][0] + 1)]
    data = data[data['eval4'] >= (dang_data['eval4'][0] - 1)]

    if(len(data) != 0):

        # 做出来应该赋值的 值
        fuzhi_num = 0
        for j in data['product_id'].unique():
            fuzhi_num += data_2448_1028[data_2448_1028['product_id'] == j]['ave_12'][data_2448_1028[data_2448_1028['product_id'] == j].index[0]]
        fuzhi_num = int(fuzhi_num / len(data))
        # 做出来可能提交的格式
        new = DataFrame()
        new['product_month'] = list(list_month)
        new['product_id'] = i
        new['ciiquantity_month'] = 0
        new.loc[new['product_month'] >= dang_data['startdate'][0] , 'ciiquantity_month'] = fuzhi_num
        new = new[['product_id','product_month','ciiquantity_month']]

        new.to_csv('../data/part3_tijiao.csv', index=False, mode='a', encoding='utf-8', header=None)


'''上面只是完成了 377 个产品，还有414-377=37 个没有-直接置0就可以了'''
list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
info_2448_1028 = pd.read_csv('../data/info_2448_1028.csv')
info_414 = pd.read_csv('../data/info_414.csv')
part3_tijiao = pd.read_csv('../data/part3_tijiao.csv' , header=None)
part3_tijiao.columns = ['product_id','month','num']
for i in part3_tijiao['product_id'].unique():
    info_414 = info_414[info_414['product_id'] != i]
print(len(info_414))
for i in info_414['product_id'].unique():
    # 先找到414个产品中  当次循环的产品数据（仅仅是1行）
    new = DataFrame()
    new['product_month'] = list(list_month)
    new['product_id'] = i
    new['ciiquantity_month'] = 0
    new = new[['product_id', 'product_month', 'ciiquantity_month']]
    new.to_csv('../data/part3_37.csv', index=False, mode='a', encoding='utf-8', header=None)















'''接下来 将两部分 用规则做出来的数据连接在一起'''

# 前面已经做了 3450个产品，现在还有550个，这550个用户可以分成2部分，第一部分是那些最大月份小于20的，但是却有数据，
# 另一部分是那些没有数据的用户
'''先做那些没有数据的用户吧，一共524个，之前已经做过了，直接拿来用就行'''
col = ['product_id','product_month','ciiquantity_month']
part3_37 = pd.read_csv('../data/part3_37.csv',header=None)
part3_37.columns = col
part3_tijiao = pd.read_csv('../data/part3_tijiao.csv',header=None)
part3_tijiao.columns = col
queshi_data = pd.read_csv('../data/queshi_data.csv',header=None)
queshi_data.columns = col
zao_data = pd.read_csv('../data/zao_data.csv',header=None)
zao_data.columns = col

part3_37 = part3_37.append(part3_tijiao)
part3_37 = part3_37.append(queshi_data)
part3_37 = part3_37.append(zao_data)

part3_37.to_csv('../data/data_524.csv', index=False, mode='a', encoding='utf-8')


'''看样子还剩 550 - 524 = 26个，直接用均值来代替吧'''
data_524 = pd.read_csv('../data/data_524.csv')
data_3450 = pd.read_csv('../data/data_3450.csv')
# 求上面的并集
id_1 = list(data_524['product_id'].unique())
id_2 = list(data_3450['product_id'].unique())
bingji = list(set(id_1).union(set(id_2)))
# 求出来差集
id_3 = range(1,4001)
chaji = list(set(id_3).difference(set(bingji))) # b中有而a中没有的
print(len(chaji)) # 26


'''在 month_num_2.csv 中找出来 这26个 产品的数据，同时用均值做出来提交数据'''
list_month = ['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01','2016-07-01',\
                '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
month_num_2 = pd.read_csv('../data/month_num_2.csv',header=None)
month_num_2.columns = ['product_id','num','month']
for i in chaji:
    data = month_num_2[month_num_2['product_id'] == i]
    ave = data['num'].mean()
    # 组建可以提交的数据
    new = DataFrame()
    new['product_month'] = list(list_month)
    new['product_id'] = i
    new['ciiquantity_month'] = ave
    new = new[['product_id', 'product_month', 'ciiquantity_month']]
    new.to_csv('../data/data_' + str(len(chaji)) + '.csv', index=False, mode='a', encoding='utf-8', header=None)


'''将3个文件组建好来提交'''
data_524 = pd.read_csv('../data/data_524.csv')
data_3450 = pd.read_csv('../data/data_3450.csv')
data_26 = pd.read_csv('../data/data_26.csv',header=None)
data_26.columns = list(data_524.columns)

data = data_524.append(data_3450)
data = data.append(data_26)

data = data.sort_values(by=['product_id', 'product_month'])
data.to_csv('../data/new_1.csv', index=False, mode='a', encoding='utf-8')





'''将其中那部分数据不完全的产品（550个）用均值134平滑下'''
part_2 = pd.read_csv('../data/new_1.csv')
for i in list(data_524['product_id'])+list(data_26['product_id']):
    part_2.loc[part_2['product_id']==i,'ciiquantity_month'] = 134

part_1 = pd.read_csv('../data/new_1.csv')
for i in part_2['product_id'].unique():
    data_1 = part_1[part_1['product_id'] == i]
    data_2 = part_2[part_2['product_id'] == i]
    data_1 = data_1.reset_index(drop=True)
    data_2 = data_2.reset_index(drop=True)
    data_1['ciiquantity_month'] = data_1['ciiquantity_month'] * 0.5 + data_2['ciiquantity_month'] * 0.5
    data_1['ciiquantity_month'] = data_1['ciiquantity_month'].astype('int')
    if(i == 1):
        data_1.to_csv('../data/new_2.txt' , index=False , mode='a' , encoding='utf-8')
    else:
        data_1.to_csv('../data/new_2.txt', index=False, mode='a', encoding='utf-8',header=None)





















