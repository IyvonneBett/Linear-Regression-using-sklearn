import pandas
import sklearn
import matplotlib.pyplot as plt
df = pandas.read_csv('Advertising.csv')
print(df)
print(df.describe())
print(df.corr())
# a correlation from 0  to 1 one goes up the the other goes up or one goes down the other goes down
# from 0 to -1 one goes up the other down
# Machine L does not work with empty records it must be filled ...does not also work with text data...must be encoded to 0s and 1s\
print(df.isnull().sum()) # no empties
print(df.dtypes) # no text, they are floats and ints
 # split the data to features and labels
array = df.values # we read all data into an array
features = array[:, 1:4] # : - all rows...4th colum is not counted here count from 0
labels = array[:,4] # 4th column which is the sales counted here

# we split 70% of data for training,,,,,140 records
# we split 30%  of  data for testing.....60 records
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.30, random_state=42)

# linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
print('Traning.....finished')

# ask the model to predict X_test
predictions = model.predict(X_test)
print(predictions) # model will predict sales from X_test....we hide y_test which has the sales

# compare the predictions and y_test(Sales)
from sklearn.metrics import  r2_score
print('R squared =', r2_score(Y_test, predictions))

from sklearn.metrics import mean_squared_error
print('mean squared error = ', mean_squared_error(Y_test, predictions))
# we need to find square root

# plotting....we visualize the model performance in a scatter plot
import matplotlib.pyplot as plt
plt.style.use('seaborn')
figure,ax = plt.subplots()
ax.scatter(Y_test, predictions, color= 'green') # scatter
ax.plot(Y_test, Y_test, color= 'red') # Best fit # line
ax.set_xlabel('Y Test')
ax.set_ylabel('Model predictions')
ax.set_title('Y Test vs Model Predictions')
plt.show()

# new expenditure
expense = [[0 ,400,0]] # tv , radio, newspaper.....radio was better 82
observation = model.predict(expense)
print('You can sell:', observation, 'Units')






