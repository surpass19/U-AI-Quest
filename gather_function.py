import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import pandas_profiling as pdp
import numpy as np

import lightgbm as lgb

#回帰の可視化
#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

import shap
# import xgboost

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


import warnings
warnings.filterwarnings('ignore')

"""=================================================="""

def count_amenity(x):
    return len(str(x).split('{')[1].split('}')[0].split(','))

def preprocessing(ori_df, train_max_amenities_list=None):
    df = ori_df.copy()
    
    """数値変数"""
    df["bathrooms"] = df["bathrooms"].fillna(0).astype('int')
    df["bathrooms_par_1"] = df["bathrooms"] / df["accommodates"]
    
    df["bedrooms"] = df["bedrooms"].fillna(0).astype('int')
    df["bedrooms_par_1"] = df["bedrooms"] / df["accommodates"]
    
    df["beds"] = df["beds"].fillna(0).astype('int')
    df["beds_par_1"] = df["beds"] / df["accommodates"]
    
    df["bed_par_bedrooms"] = df["beds"] / df["bedrooms"]
    
    df["latitude_int"] = df["latitude"].astype(int)
    df["longitude_int"] = df["longitude"].astype(int)
    
    df["latitude_longitude"] = df["latitude"] * df["longitude"]
    
    df["review_score_total"] = df["review_scores_rating"].fillna(0) * df["number_of_reviews"].fillna(0)
    df["review_score_weight"] = df["review_scores_rating"].fillna(0) / (df["number_of_reviews"].fillna(0) + 100)
    
    
    #アメニティ
    df['amenities_count'] = df["amenities"].map(count_amenity)
    df['amenities_type'] = df['amenities'].apply(lambda x: x.replace('{', '').replace('}', '').split(','))
    index = df.sort_values('amenities_count', ascending=False).index[0]
    max_amenities_list = df['amenities_type'][index]
    
    
    if train_max_amenities_list != None:
        #もし第2引数があれば, max_amenities_listをそっちに更新
        max_amenities_list = train_max_amenities_list
        print('max_amenities_list : 更新')
    
    for amenity in tqdm(max_amenities_list):
        amenity = amenity.replace('"', '')
        #print(amenity)
        df[f'{amenity}_add'] = df['amenities'].str.contains(amenity).astype(str)
        
    df['rare_amenities_count'] = df["amenities"].apply(lambda x: x.count('"'))
    
    #description
    df['description_word_count'] = df['description'].map(lambda x: len(x))
    
    #返信率
    host_response_rate = df['host_response_rate'].str.split("%", expand=True)[0]
    df['host_response_rate'] = host_response_rate.fillna(0).astype('int')
    df['host_response_rate_weight'] = df['host_response_rate'] * 100 / (df["number_of_reviews"].fillna(0) + 100)
    
    #最初のレビュー日
    df['first_review'] = pd.to_datetime(df['first_review'])
    df["first_review_Year"] = df["first_review"].apply(lambda x:x.year)
    df["first_review_Month"] = df["first_review"].apply(lambda x:x.month)
    df["first_review_Day"] = df["first_review"].apply(lambda x:x.day)
    df["kijun"] = "2008-11-17"
    df["kijun"] = pd.to_datetime(df["kijun"])
    df["BusinessOld"] = (df["first_review"] - df["kijun"]).apply(lambda x: x.days)
    
    #登録した日
    df['host_since'] = pd.to_datetime(df['host_since'])
    df["host_since_Year"] = df["host_since"].apply(lambda x:x.year)
    df["host_since_Month"] = df["host_since"].apply(lambda x:x.month)
    df["host_since_Day"] = df["host_since"].apply(lambda x:x.day)
    df["kijun2"] = "2008-3-3"
    df["kijun2"] = pd.to_datetime(df["kijun2"])
    df["BusinessOld2"] = (df["host_since"] - df["kijun2"]).apply(lambda x: x.days)
    df["first_reviewOld"] = (df["first_review"] - df["host_since"]).apply(lambda x: x.days)
    
    #最後のレビュー日
    df['last_review'] = pd.to_datetime(df['last_review'])
    df["last_review_Year"] = df["last_review"].apply(lambda x:x.year)
    df["last_review_Month"] = df["last_review"].apply(lambda x:x.month)
    df["last_review_Day"] = df["last_review"].apply(lambda x:x.day)
    df["kijun3"] = "2021-12-30"
    df["kijun3"] = pd.to_datetime(df["kijun3"])
    df["BusinessOld3"] = (df["last_review"] - df["kijun"]).apply(lambda x: x.days)
    df["BusinessUpdate"] = (df["kijun3"] - df["last_review"]).apply(lambda x: x.days)
    df["BusinessPeriod"] = (df["last_review"] - df["first_review"]).apply(lambda x: x.days)
    
    df['thumbnail_url'][df['thumbnail_url'].notnull()] = 1
    df['thumbnail_url_str'] = df['thumbnail_url'].astype(str)
    
    
    zipcode = df['zipcode'].str.replace('Near', '').str.replace('m', '').replace(' ', np.nan)\
                                                                                 .str.split("\r", expand=True)[0].str.split("-", expand=True)[0].str.split(".", expand=True)[0]
    df['zipcode_int'] = zipcode
    df['zipcode_int'] = df['zipcode_int'].fillna(0).astype(int)
    
    
    #不必要なcolumnを落とす
    df.drop(['amenities'], axis=1, inplace=True)
    df.drop(['amenities_type'], axis=1, inplace=True)
    df.drop(['description'], axis=1, inplace=True)
    df = df.drop('first_review', axis=1)
    df = df.drop('kijun', axis=1)
    df = df.drop('host_since', axis=1)
    df = df.drop('kijun2', axis=1)
    df = df.drop('last_review', axis=1)
    df = df.drop('kijun3', axis=1)
    df = df.drop('zipcode', axis=1)
    
    #特徴量にできてない
    #df = df.drop('id', axis=1)
    df = df.drop('name', axis=1)
    
    #inf消す
    df=df.replace([np.inf, -np.inf], 0)
    
    #型変更
    cat_col = [col for col in df.select_dtypes(include=object)]
    num_col = [col for col in df.select_dtypes(exclude=object)]
    df[cat_col] = df[cat_col].fillna('Na')
    
    return df, max_amenities_list

"""=================================================="""

#説明変数を対数変換
def logarithmic_transformation(df):
    num_col = [col for col in df.select_dtypes(exclude=object)]
    
    #各説明変数の歪度を計算
    skewed_feats = df[num_col].apply(lambda x: x.skew()).sort_values(ascending = False)
    
    
    #歪度の絶対値が0.5より大きい変数だけに絞る
    skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5]
    
    #欠損値のないものに絞る
    num_col_feat_list = []
    for i in skewed_feats_over.index:
        flag = df[i].isnull().any()
        if not flag:
            num_col_feat_list.append(i)

    print(num_col_feat_list)
    
    #グラフ化
    skewed_feats_over_plot = skewed_feats_over[num_col_feat_list]
    
    plt.figure(figsize=(20,10))
    plt.xticks(rotation='90')
    sns.barplot(x=skewed_feats_over_plot.index, y=skewed_feats_over_plot)
    
    #Yeo-Johnson変換
    pt = PowerTransformer()
    pt.fit(df[num_col_feat_list])

    #変換後のデータで各列を置換
    tmp = pd.DataFrame()
    tmp[num_col_feat_list] = pt.transform(df[num_col_feat_list])
    tmp = tmp.add_prefix('Log_')
    df[tmp.columns] = tmp
    
    #各説明変数の歪度を計算
    skewed_feats_fixed = df[tmp.columns].apply(lambda x:x.skew()).sort_values(ascending = False)

    #グラフ化
    plt.figure(figsize=(20,10))
    plt.xticks(rotation='90')
    sns.barplot(x=skewed_feats_fixed.index, y=skewed_feats_fixed)

    return df, num_col_feat_list, pt

"""=================================================="""

#カテゴリカル変数化
def process_categorical(df, target_columns):
    df2 = df.copy()
    for column in target_columns:
        df2[column] = LabelEncoder().fit_transform(df2[column].fillna('Na'))

    #ターゲットカラム以外にカテゴリ変数があれば, ダミー変数にする
    df2 = pd.get_dummies(df2, drop_first=True)

    for column in tqdm(target_columns):
        df2[column] = df2[column].astype('category')

    return df2

"""=================================================="""

#予測値と正解値を描写する関数
def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    R2 = r2_score(pred_df['true'], pred_df['pred'])
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min()-0.05 , pred_df.max().max()+0.05)
    ax.set_ylim(pred_df.min().min()-0.05 , pred_df.max().max()+0.05)
    x = np.linspace(pred_df.min().min()-0.05, pred_df.max().max()+0.05, 2)
    y = x
    ax.plot(x,y,'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)