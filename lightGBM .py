#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import lightgbm as lgb
from google.colab import drive
drive.mount('/content/drive')

# In[126]:
res_feature = [
  # 'SHH_BCK',
 'REG_CPT',
#  'MON_12_EXT_SAM_TRSF_IN_AMT',
#  'AGN_CNT_RCT_12_MON',
#  'HLD_FGN_CCY_ACT_NBR',
#  'MON_12_EXT_SAM_AMT',
 'LAST_12_MON_MON_AVG_TRX_AMT_NAV',
 'AGN_AGR_LATEST_AGN_AMT',
 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
#  'MON_12_EXT_SAM_NM_TRSF_OUT_CNT',
#  'MON_12_TRX_AMT_MAX_AMT_PCTT',
#  'NB_RCT_3_MON_LGN_TMS_AGV',
#  'MON_12_ACM_ENTR_ACT_CNT',
#  'MON_12_AGV_ENTR_ACT_CNT',
#  'MON_12_EXT_SAM_TRSF_OUT_AMT',
#  'HLD_DMS_CCY_ACT_NBR',
#  'MON_12_ACM_LVE_ACT_CNT',
#  'MON_12_AGV_LVE_ACT_CNT',
#  'MON_12_AGV_TRX_CNT',
#  'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY',
#  'MON_12_ACT_IN_50_UP_CNT_PTY_QTY',
#  'CUR_MON_COR_DPS_MON_DAY_AVG_BAL',
#  'AGN_CUR_YEAR_WAG_AMT',
#  'MON_6_50_UP_ENTR_ACT_CNT',
#  'MON_6_50_UP_LVE_ACT_CNT',
 'LAST_12_MON_COR_DPS_DAY_AVG_BAL',
#  'REG_DT',
 'COR_KEY_PROD_HLD_NBR',
#  'AGN_CUR_YEAR_AMT',
#  'COUNTER_CUR_YEAR_CNT_AMT',
#  'PUB_TO_PRV_TRX_AMT_CUR_YEAR',
#  'CUR_YEAR_COUNTER_ENCASH_CNT',
 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
#  'OPN_TM',
#  'CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT',
#  'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT',
#  'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
#  'EMP_NBR',
#  'CUR_YEAR_MON_AGV_TRX_CNT',
#  'ICO_CUR_MON_ACM_TRX_TM',

 'LGP_HLD_CARD_LVL_A',
 'LGP_HLD_CARD_LVL_B',
 'LGP_HLD_CARD_LVL_C',
 'LGP_HLD_CARD_LVL_D',
 'LGP_HLD_CARD_LVL_E',
 'LGP_HLD_CARD_LVL_F',
#  'MON_12_CUST_CNT_PTY_ID_Y',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_A',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_B',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_C',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_D',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_E',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_F',
#  'WTHR_OPN_ONL_ICO',
 'WTHR_OPN_ONL_ICO_A',
 'WTHR_OPN_ONL_ICO_B'
 ]


# In[34]:

train_data = pd.read_feather('/content/drive/MyDrive/colab/FinTech/train_data_B_2.feather')
test_data = pd.read_feather('/content/drive/MyDrive/colab/FinTech/test_data_B_2.feather')
train_data.head()


# In[35]:


test_data.head()


# In[36]:


# 处理行缺失
import math
def showNan(row):
  flag = 0
  for i in row[1:]:
    if math.isnan(i) or i == 2:
      flag += 1
  if flag > 28:
    return False 
  return True 
train_data = train_data.loc[train_data.apply(showNan, axis=1)].reset_index(drop=True)
train_data.shape


# In[37]:


# 特征筛选
def feature_select_pearson(train, test, k):
    features = [f for f in train.columns if f not in ['CUST_UID', 'label']]
    featureSelect = features[:]
    for fea in features:
        if train[fea].isnull().sum() / train.shape[0] >= 0.9:
            featureSelect.remove(fea)
    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'label']].fillna(0).corr(method='spearman').values[0][1]))
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    return se[:].index.tolist()


# In[38]:


features = feature_select_pearson(train_data, test_data, 60)
san_features = [
 'LGP_HLD_CARD_LVL_A',
 'LGP_HLD_CARD_LVL_B',
 'LGP_HLD_CARD_LVL_C',
 'LGP_HLD_CARD_LVL_D',
 'LGP_HLD_CARD_LVL_E',
 'LGP_HLD_CARD_LVL_F',
 'MON_12_CUST_CNT_PTY_ID_Y',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_A',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_B',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_C',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_D',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_E',
 'NB_CTC_HLD_IDV_AIO_CARD_SITU_F',
 'WTHR_OPN_ONL_ICO',
 'WTHR_OPN_ONL_ICO_A',
 'WTHR_OPN_ONL_ICO_B']
lian_features = [x for x in features if x not in san_features]
# 处理极大值
filters = []
for i in lian_features:
  if train_data[i].nunique() > 10000:
    filters.append((i, train_data[i].max()))
filters = [x[1] for x in filters]
def filters_apply(row):
  for i in row:
    if i in filters:
      return False 
  return True 
train_data = train_data.loc[train_data.apply(filters_apply, axis = 1)]
train_data.shape


# In[39]:


from sklearn.model_selection import KFold
params = {
    'boosting_type': 'gbdt',  # boosting方式
    'objective': 'binary',  # 任务类型为「多分类」
    'max_depth': 7,
    'num_leaves': 127,  # 最大的叶子数
    'feature_fraction': 0.8,  # 原来是0.8
    'bagging_fraction': 0.8,  # 原来是0.8
    'bagging_freq': 5,  # 每5次迭代，进行一次bagging
    'learning_rate': 0.013,  # 学习效率：原来是0.1
    'seed': 2,  # seed值，保证模型复现                   #
    'n_jobs': 4,  # 多线程
    'verbose': 1,
    #  ‘device’ : ‘gpu’, ‘gpu_platform_id’:0, ‘gpu_device_id’:0
    'metric': 'auc',
    'lambda_l1': 0.7,  # 新添加 L1
    'lambda_l2': 0.7,  # 新添加 L2
#     'min_data_in_leaf': 20,  # 叶子可能具有的最小记录数
}


# In[129]:


from sklearn.metrics import roc_auc_score
def train_predict(train, test, params, k):
    prediction_test = list()
    label = 'label'
    features = [f for f in train.columns if f not in ['CUST_UID', 'label']]
    params = params 
    tf = KFold(n_splits=k, random_state=2022, shuffle=True)
    cv_score = []
    prediction_train = pd.Series()
    ESR = 200
    NBR = 1000
    VBE = 100
    for train_part_index, eval_index in tf.split(train[features], train[label]):
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                train[label].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                                train[label].loc[eval_index])
        bst = lgb.train(params, train_part, num_boost_round=NBR,
                       valid_sets=[train_part, eval],
                       valid_names=['train', 'valid'],
                       early_stopping_rounds=ESR, verbose_eval=VBE)
        prediction_test.append(bst.predict(test[features]))
        eval_pre = bst.predict(train[features].loc[eval_index])
        score = roc_auc_score(train['label'].loc[eval_index], eval_pre)
        cv_score.append(score)
    print(cv_score, sum(cv_score)/k)
    test['label'] = np.array(prediction_test).mean(axis=0)
    test['CUST_UID'] = test_data['CUST_UID']
#     test['label'] = prediction_test[-1]
    test[['CUST_UID', 'label']].to_csv('/content/drive/MyDrive/colab/FinTech/lgb_flnal_submit2.txt', header = None, index = None, sep='\t')
    return sum(cv_score)/k


# In[41]:


# curmax = 37 0.88444
# 五折
# 38  8843
# 39  8843
# 40  88492515

# 原始数据
# 五折 8730


# In[42]:


train_data.reset_index(drop=True)


# In[130]:


# train_predict(train.iloc[:,1:], test.iloc[:,1:], params, 5)
train_predict(train_data[res_feature + ['label','CUST_UID']].reset_index(drop=True), test_data[res_feature + ['CUST_UID']], params, 5)

