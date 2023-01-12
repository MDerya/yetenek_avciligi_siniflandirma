


#       Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma


#       İş Problemi
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre,
# oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.



#       Veri Seti Hikayesi

#   Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların,
#   maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

# scoutium_attributes.csv
# 8 Değişken 10.730 Gözlem

# task_response_id Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id      İlgili maçın id'si
# evaluator_id  Değerlendiricinin(scout'un) id'si
# player_id     İlgili oyuncunun id'si
# position_id   İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#               1: Kaleci
#               2: Stoper
#               3: Sağ bek
#               4: Sol bek
#               5: Defansif orta saha
#               6: Merkez orta saha
#               7: Sağ kanat
#               8: Sol kanat
#               9: Ofansif orta saha
#               10: Forvet
# analysis_id   Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id  Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
#

# scoutium_potential_labels.csv
# 5 Değişken 322 Gözlem

# task_response_id Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id        İlgili maçın id'si
# evaluator_id    Değerlendiricinin(scout'un) id'si
# player_id       İlgili oyuncunun id'si
# potential_label Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)



####################### GÖREVLER ###################################
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler #standartlastırma, dönüştürme fonksiyonları
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Adım1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

df_a = pd.read_csv("scoutium_attributes.csv", sep=";")
df_p = pd.read_csv("scoutium_potential_labels.csv", sep=";")
df_a.head()
df_p.head()

df_a.columns

# Adım2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

df= pd.merge(df_p, df_a, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="left")
df.head()

# Adım3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df["position_id"].unique()
df.shape
df = df[df["position_id"] != 1]
#ya da
#df= df[~(df["position_id"] == 1)]


# Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df["potential_label"].unique()
df.shape
df= df[~(df["potential_label"] == "below_average")]
#ya da
# df = df[df["potential_label"] != "below_average"]

# Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.


#   Adım1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#   “attribute_value” olacak şekilde pivot table’ı oluşturunuz.


#                                                   4322        4323        4324        4325        4326        4327     ..
# player_id     position_id     potential_label ..
# 1355710       7               average             50.500      50.500      34.000      50.500      45.000      45.000  ..
# 1356362       9               average             67.000      67.000      67.000      67.000      67.000      67.000  ..
# 1356375       3               average             67.000      67.000      67.000      67.000      67.000      67.000  ..
#               4               average             67.000      78.000      67.000      67.000      67.000      78.000  ..
# 1356411       9               average             67.000      67.000      78.000      78.000      67.000      67.000  ..



table = pd.pivot_table(df, index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")
table.head()

#  Adım2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

table.reset_index(inplace=True) #ya da table = table.reset_index(drop=False)
table.head()
table.dtypes
#“attribute_id” sü-tun-la-rı-nın i-sim-le-ri-ni stringe cevirelim
table.columns = table.columns.map(str)



# Adım6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
df["potential_label"].unique()
df["potential_label"].head()
le = LabelEncoder() #LabelEncoder nesnemizi getiriyoruz

le.fit_transform(df["potential_label"])[0:5] #ilk 5ine bakalım
#bu nesneyi fit_transform metodunu kullanarak "potential_label" degiskenine uyguluyoruz.
# !!Alfabetik sıraya göre ilk gördügüne 0 degerini verir,digerine 1 - average=0 highlighted=1

le.inverse_transform([0, 1]) #diyelim ki hangisi 0 hangisi 1 unuttuk,bunu tespit etmek icin

#label_encoder fonk yazalım
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(table,"potential_label")


# Adım7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.

table.info()
table.head()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(table)

num_cols

# Adım8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

# StandardScaler: Klasik standartlaştırma(normalleştirme).
# Bütün gözlem birimlerinden Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s


scaler = StandardScaler()
table[num_cols] = scaler.fit_transform(table[num_cols])

table[num_cols].head()

# Adım9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

#Bagımlı ve bagımsız degiskenlerimizi ayırıyoruz
y = table["potential_label"]
X = table.drop(["potential_label", "player_id"], axis=1)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
                   ("LightGBM", LGBMClassifier())]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

#  Hiperparametre Optimizasyonu yapınız.


lgbm_model = LGBMClassifier(random_state=46)

#rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# normal y cv süresi:
# scale edilmiş y ile:

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))


# Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

# Degiskenlerin önem düzeyini belirten feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)

df.head()