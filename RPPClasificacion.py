import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from keras.models import load_model
from multiprocessing import Pool, freeze_support

TRAIN_SIZE = 0.8
SEQUENCE_LENGTH = 300
EPOCHS = 10
BATCH_SIZE = 1024
MAX = 4521

tokenizer = Tokenizer()

df = pd.read_csv("datos.csv")

df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)

tokenizer.fit_on_texts(df_train.text_clean.astype(str))

# KERAS

model = load_model('model.h5')
noticiasRPP = pd.read_excel("C:/Users/jazmi/Documents/TCG/Modelo DL/NUEVO/nuevas_noticias.xlsx")

"""
noticiasRPP = pd.read_excel("C:/Users/jazmi/Documents/TCG/RPP/noticias_rpp.xlsx")
print(noticiasRPP.shape)
noticiasRPP['Departamento'].replace('', np.nan, inplace=True)
noticiasRPP.dropna(subset=['Departamento'], inplace=True)
noticiasRPP['Salud'] = ""
print(noticiasRPP.shape)
"""



def decode_sentiment(score):
    if score < 0.5:
        return "NEGATIVE"
    else:
        return "POSITIVE"


def predict1(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score)

    print(label)
    return label


def predict2(idx):
    return predict1(noticiasRPP['Titular'].iloc[idx])


'''
pool = mp.Pool(mp.cpu_count())
print(pool)
results = pool.map(predict2, [i for i in range(0, 5)])

pool.close()
print("\n ---------------\n ")
print(len(results))
print("\n ---------------\n ")
'''


def fun1(min1):
    max1 = min1 + MAX
    resu = list()
    for j in range(min1, max1):
        s = noticiasRPP['Titular'].iloc[j]
        print(s)
        res = predict1(s)
        resu.append(j)
        resu.append(res)

    #print(resu)
    return resu


def main():
    pool = mp.Pool(mp.cpu_count())
    print(pool)
    results = pool.map(fun1, [i*MAX for i in range(0, 12)])
    return results



if __name__ == "__main__":
    freeze_support()
    res = main()
    print(res)
    clasificacion = np.concatenate(res)


    for r in range(0, len(clasificacion) - 1, 2):

        noticiasRPP['Salud'][int(clasificacion[r])] = clasificacion[r + 1]



    noticiasRPP.to_excel("noticiasRPPClasificadas.xlsx")