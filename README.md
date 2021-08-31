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


## 他の方の解法
* スコア141.76
モデル LGBMを採用しました。CatBoostやニューラルネットワークも試してみましたが、LGBMの方が高かったです。
前処理 : カテゴリカルデータ(テキストデータを除く) LabelEncodingとTargetEncodingを適用しました。xfeatとというライブラリを使用すれば簡単に実装できます。
数値データ : 色々除算を行いました。
テキストデータ : tf-idfベクトルに変換した後、SVDで50次元に削減しました。
日付データ : 年、月、日、曜日の数値データに変換しました。
その他 : ZipCodeとRoomtypeをキーとしてbathrooms × bedsroomsとaccommodatesの集約特徴量を求めました。
学習はfold数15のCrossValidationで行い乱数シードを変えて3つのLightGBMモデルを作成しました。その後、Ridge回帰を用いてstackingを行いました。


* スコア143.8756549
データ処理
 - latitude, longitude
    >> 2次元に直し，グリッドに区切ったものを説明変数に追加
 - amenities
    >> 各アメニティ(TV,Washer..)の有無を0,1で表現し，それぞれを説明変数に追加 (かなりの数になりました)
    >> 文字数,アメニティの個数を説明変数に追加
 - property_type
    >> 出現数の少ない群は other に書き換え
 - thumbnail_url
    >> 有無を0,1で表現し，説明変数に追加
 - description
    >> 文字数を説明変数に追加
モデル作成
 - lightGBMを使用
 - 訓練データの分割方法を変え，数十個のモデルを作成
 - 訓練時のスコアの上位/下位のモデルを省き，残りのモデルの平均値を回答とした

* スコア：144.285, 順位：30位
前処理：
- accommodates: 影響大きそうだったのでlog(x), exp(x), x^2と適当に変換した変数を追加(根拠なし)
- ammenities: ダミー変数化(130個くらい変数追加) & ダミー変数の1を合計した変数を追加(アメニティの数)
- description, name: 文字列の長さを持つ変数を追加
- first_review, host_since, last_review: 最小値を起点とした日数を示す変数を追加(importanceを見るとこれが結構効いたかも)
- first_review, host_since, last_review: 日付を年ごとに1-3月、4-6月、7-9月、10-12月の4期に分けた変数を作成(例：2016の1期=>20161という数値を持つ変数)
- zipcode: 最初の1桁、2桁、3桁、5桁までの数値を持つ変数をそれぞれ追加
- その他：欠損値処理
モデル：
lightGBM+optunaでKFold=5のCVを実施
random_stateを変えて5回分のスコアの平均を使用

* スコア：145.39, 順位：50位
モデル：xgbとLGBM
前処理：
amenities ：リストではなくカンマで区切られた文字列だったので、文字列内のカンマの数＋１＝アメニティ数としました。アメニティ数多い→まあまあ人気の宿→宿泊価格は高い？　と考えました。
bed_type、city、property_type、zipcode ：CountEncodingしました。
cancellation_policy ：厳しい順に0~4の数値にしました。欠損値は-1にしました。
description ：文字数に変換しました。（文字数が長い→宿オーナーが頑張って宿泊して欲しいアピールしている→何もアピールしないよりは宿泊の人気に寄与しているはず→宿泊費は高くなる（？）　と考えました。）
last_review ：2017/10/05から最終レビュー日間の日数を求めて、「最終レビューからの経過日数」という特徴量を作成しました。
latitude ：小数点第2位まで丸めました。
longitude ：小数点第2位まで丸めました。
number_of_reviews ：number of reviews / 最終レビューからの経過日数*100 で「人気度」という特徴量を定義しました。
モデル：
検証器の中でxgbとLGBMを用いて予測値を出し、両者をどのくらいの比率で足せば目的変数に近づくかをExcelのソルバーで比率を求め、testデータの予測値にもその比率を適用させました。xgbの方が良く効いてた気がします。
振り返り：
amenities個数を算出するコードの書き方は、我ながら上手かったと思っています(笑)。amenitiesの中身をもっと深堀りすべきだったと思います。Wifiがない→人気がない宿or観光地が近くにない→宿賃も安い？など工夫できたかと思います。
時系列コンペほど時間軸の概念が大事ではなかったので、日付の三角関数化は不要だったかと思いました。
宿泊先の位置情報は、緯度経度を粗めにとればzipcodeや旅行先の目的地も概ね一致するのでは？と思いましたが、上手く使えたかわかりません…。
nameをどう使うべきか悩みました。文字数化する、頻出単語を拾うなども考えましたが、エアプのまま終わりました。(笑)


* スコア146.898, 順位103位
PreProcess（前処理）:
LightGBMで扱えるようにするところまでをスコープとして、ざっくり4工程で前処理しました。
「amenities」と「amenities以外の項目」で分ける
amenitiesだけが配列構造だったので、別テーブルから無理矢理つなげたデータと見なしました。
amenitiesを処理する
amenitiesの項目を分解して、one-hotの項目にする
one-hotのそれぞれの列で集計して、偏りが大きすぎると判断したものを消す
妥当かどうかわかりませんが、True/Falseのどちらかが全体の5%未満だったら消しています。
amenities以外の項目から、object型のカラムをLightGBMで扱える形に変換する
数値化できるものを数値化する。
t/fとなっているカラムをbool化する
nullが含まれる項目はとりあえず1.0/0.0/NaNのfloat化してみましたが、意味なかった（というかまずい？）かも。
日付カラムをUNIX秒に変換する
数値化できなかったものを処理する
カーディナリティが高いものと、そうでないものに分けて処理する
カーディナリティが高いものについて、基本は項目削除し、「有/無」に意味があるものは拾う
サムネイルや記述の登録があるかどうかはホストとしての品質、誠実性の指標になると判断して、「ちゃんと記入したか否か」、つまりnullか否かだけの値にしました。
それ以外は地理的データ含めてdropしています。
カーディナリティが低いものを尺度指標とそれ以外に分けて処理する
cancellation_policyは尺度っぽいので数値でリマップしました。
それ以外はdf[column_name].astype('category')でカテゴリ型化しています。
「2.」,「3.」のデータを連結する
id列は連結後dropしています。

* スコア：148.7053728, 順位：195位
アメニティについては各項目を分解し、TV, Cable TV や Internet, Wi-Fi などはグループ化してTV有無、ネット有無にしたり
建物の設備やセキュリティ設備などは個数をカウントなどして説明変数に使いました。
また、レビュー数が少ないものはレビュースコアを平均値に置換するなどしました。
モデルは lightGBM を使用しました。


* スコア159.69, アセスメント561位
箱ひげ図や相関係数でデータ分析を行い、価格に影響するコラムを以下に絞りました。
select_columns = ['accommodates','bathrooms','bed_type','bedrooms','beds','cancellation_policy','city','cleaning_fee','host_response_rate','number_of_reviews','review_scores_rating','room_type']
欠損値処理は、定量値は平均を、定性値は最頻値を入れました。
あとは標準化、ダミー変数の作りの工程を経ました。
モデル作りはXGBoostが人気と聞いて、それを適用しました。
疑問なのが、与えられたデータを7:3に分けて7割をTrainとしてモデルを作り、3割TestにしたものでRMSEを測ったら120前後だったんですよね。
分け方のパターンを変えても同じくらいの値なので過学習ではないと思ったのですが、提出した結果のMSEは159だったので、ここの差はなんだろうというのが疑問です。


* スコア163.74, アセスメント622位
amenities,description,name → 文字数をカウント
first_review,last_review,host_since → ある地点を1とし、それから何日か、という方法で数値化
thumbnail_url → 欠損値を0、それ以外を1に変換しました。
その他の欠損値 → ブール値の欠損値はfalse,数値の欠損値は中央値で埋めました。
host_responce_rate → ％を除去し、数値
ダミー化
PLSでモデル化
長文の説明変数は単語に分けて処理したり、zipcodeをもう少しきれいにそろえたり、中央値ではなく平均値で埋めたり、
モデル化を何回も行って結果をアセンブルしたり、出来ることはいっぱいあったなと、反省点は多々です。

* スコア: 163.8393301, 順位: 624位
accommodates, bathrooms, bed_type, bedrooms, beds, city, cleaning_fee, host_has_profile_pic, host_identity_verified, instant_bookable, latitude, longitude, number_of_reviews, property_type, room_type
を分析に利用しました。
同一zipcodeにおける価格の平均値（いわゆる近隣相場価格）
host_sinceを基に設置からの経過月数を算出し、設置からの経過月数が同じかつnumber_of_reviewsが同じ物件の価格の平均値（airbnbはレビューが価格に影響する、というWeb情報を基に考えました）




