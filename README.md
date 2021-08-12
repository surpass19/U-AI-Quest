# U-AI-Quest

## model-hold2
validデータ　✖️<br>
* RMSE : 104.19 <br>
* R2 : 0.62

## model-hold3
testデータ　<br>
* RMSE : 106.45 <br>
* R2 : 0.604

## model-khold3
testデータ　<br>
* RMSE : 105.62 <br>
* R2 : 0.61


## model-hold3 ⇨ model-hold-submit
PB 168.48

## model-khold3-2 (提出)
testデータ　<br>
* RMSE : 106.9 <br>
* R2 : 0.59
* PB 160.48

## model-khold3-2
testデータ　<br>
logあり　<br>
Params: 
    objective: regression<br>
    metric: rmse<br>
    random_seed: 0<br>
    feature_pre_filter: False<br>
    lambda_l1: 0.24341699147974324<br>
    lambda_l2: 8.890301958197547<br>
    num_leaves: 31<br>
    feature_fraction: 0.4<br>
    bagging_fraction: 1.0<br>
    bagging_freq: 0<br>
    min_child_samples: 10<br>
 PB：179.2199942

## 初めて知った
説明・目的変数の対数化

## もっとやればよかった？
自然言語の扱い
年代や月のsin化
PCA
いろんなモデル試す
