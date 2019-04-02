import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')
df.head(5)
df['test']= df['parent_comment'] #+" " + df['parent_comment']
df.head(3)

#remove numbers
df['test'] = df['test'].str.replace('\d+', '')
#remove punctuations
df['test'] = df['test'].str.replace('[^\w\s]','')

# Removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['test'] = df['test'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['test']= df['test'].apply(lambda x: x.lower())
df.head(3)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df['test'], df['score'], random_state = 0, test_size = 0.4)
print(len(train_x))
print(len(train_y))
print(len(test_y))
print(len(test_y))
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils
max_words = 5000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_x) # only fit on train
x_train = tokenize.texts_to_matrix(train_x)
x_test = tokenize.texts_to_matrix(test_x)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', train_y.shape)
print('y_test shape:', test_y.shape)
from sklearn import datasets, linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, train_y)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

from sklearn import metrics

# calculate MAE, MSE, RMSE
print(metrics.mean_absolute_error(test_y, y_pred))
print(metrics.mean_squared_error(test_y, y_pred))
print(np.sqrt(metrics.mean_squared_error(test_y, y_pred)))


print(y_pred.shape)

test_df = pd.read_csv('test.csv')
test_df.head()
test_df['test']= test_df['parent_comment']#+" " + test_df['parent_comment']

#remove numbers
test_df['test'] = df['test'].str.replace('\d+', '')

#remove punctuations
test_df['test'] = df['test'].str.replace('[^\w\s]','')

# Removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
test_df['test'] = test_df['test'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test_df['test']= test_df['test'].apply(lambda x: x.lower())
X = test_df['test']
tokenize.fit_on_texts(X) # only fit on train
X = tokenize.texts_to_matrix(X)
y_pred_test = regr.predict(X)
test_pred = pd.DataFrame(y_pred_test)
test_pred.to_csv('test_pred.csv')

