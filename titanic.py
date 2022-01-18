import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_Rows',None)

df = sns.load_dataset("titanic")
df.head()

df2 = df[['survived','pclass','age','parch']]
df2.head( )

df3  = df2.fillna(df2.mean())
df3.head()

X = df3.drop("survived", axis= 1)
y = df3["survived"]
print('shape of X =', X.shape)
print('shape of y =', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 51)
print('Shape of X_train =', X_train.shape)
print('Shape of X_test =', X_test.shape)
print('Shape of y_train =', y_train.shape)
print('Shape of y_test =', y_test.shape)

#StandardScaler

sc = StandardScaler()
sc.fit(X_train)

sc.mean_

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

df3.keys()
#Index(['survived', 'pclass', 'age', 'parch'], dtype='object')

X_train_sc_df = pd.DataFrame(X_train_sc, columns=['pclass', 'age', 'parch'])
X_test_sc_df = pd.DataFrame(X_test_sc, columns=['pclass', 'age', 'parch'])

X_train_sc_df.head()
X_test_sc_df.describe().round(2)

	"""pclass	age	parch
count	179.0	179.0	179.0
mean	0.0	0.0	-0.0
std	1.0	1.0	1.0
min	-2.0	-2.0	-0.0
25%	-0.0	-0.0	-0.0
50%	1.0	0.0	-0.0
75%	1.0	0.0	-0.0
max	1.0	4.0	7.0"""
#Normalization
mnc = MinMaxScaler()
mnc.fit(X_train)

X_train_mmc = mnc.transform(X_train)
X_test_mmc = mnc.transform(X_test)

X_train_mmc_df = pd.DataFrame(X_train_mmc, columns=['pclass', 'age', 'parch'])
X_test_mmc_df = pd.DataFrame(X_test_mmc, columns=['pclass', 'age', 'parch'])

X_test_mmc_df.describe().round()
"""	pclass	age	parch
count	179.0	179.0	179.0
mean	2.0	6.0	0.0
std	0.0	2.0	0.0
min	1.0	1.0	0.0
25%	1.0	5.0	0.0
50%	2.0	6.0	0.0
75%	2.0	7.0	0.0
max	2.0	15.0	1.0"""
