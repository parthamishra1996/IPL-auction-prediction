
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn import metrics
# Reading processed data

batsman_2017 = pd.read_csv("ProcessedData/batsman_2017.csv")
batsman_2016 = pd.read_csv("ProcessedData/batsman_2016.csv")
batsman_2015 = pd.read_csv("ProcessedData/batsman_2015.csv")
batsman_2014 = pd.read_csv("ProcessedData/batsman_2014.csv")

baller_2017 = pd.read_csv("ProcessedData/baller_2017.csv")
baller_2016 = pd.read_csv("ProcessedData/baller_2016.csv")
baller_2015 = pd.read_csv("ProcessedData/baller_2015.csv")
baller_2014 = pd.read_csv("ProcessedData/baller_2014.csv")

allrounder_2017 = pd.read_csv("ProcessedData/allrounder_2017.csv")
allrounder_2016 = pd.read_csv("ProcessedData/allrounder_2016.csv")
allrounder_2015 = pd.read_csv("ProcessedData/allrounder_2015.csv")
allrounder_2014 = pd.read_csv("ProcessedData/allrounder_2014.csv")


# In[29]:


# Model building

from scipy import stats

## Model1: 

## Setting training data
training_batsman = pd.concat([batsman_2014, batsman_2015, batsman_2016], ignore_index = True)
training_baller = pd.concat([baller_2014, baller_2015, baller_2016], ignore_index = True)
training_allrounder = pd.concat([allrounder_2014, allrounder_2015, allrounder_2016], ignore_index = True)

training_batsman = training_batsman[training_batsman['Price_adjusted'] > 0].reset_index()
training_baller = training_baller[training_baller['Price_adjusted'] > 0].reset_index()
training_allrounder = training_allrounder[training_allrounder['Price_adjusted'] > 0].reset_index()

training_batsman_y = training_batsman['Price_adjusted']
training_batsman_x = training_batsman.drop(['Price', 'Player', 'Price_adjusted', 'Price_score', 'Score', 'Team_t20', 'Team_odi'], axis = 1)
training_baller_y = training_baller['Price_adjusted']
training_baller_x = training_baller.drop(['Price', 'Player', 'Price_adjusted', 'Price_score', 'Score', 'Team_t20', 'Team_odi', 'BBI_t20', 'BBI_odi'], axis = 1)
training_allrounder_y = training_allrounder['Price_adjusted']
training_allrounder_x = training_allrounder.drop(['Price', 'Player', 'Price_adjusted', 'Price_score', 'Score', 'Team_t20', 'Team_odi', 'BBI_t20', 'BBI_odi'], axis = 1)

training_batsman.fillna(0,inplace=True)
training_baller.fillna(0,inplace=True)
training_allrounder.fillna(0,inplace=True)

## Setting test data
test_batsman = batsman_2017
test_baller = baller_2017
test_allrounder = allrounder_2017

test_batsman = test_batsman[test_batsman['Price_score'] > 0].reset_index()
test_baller = test_baller[test_baller['Price_score'] > 0].reset_index()
test_allrounder = test_allrounder[test_allrounder['Price_score'] > 0].reset_index()
test_batsman['Price_adjusted'] = test_batsman['Price_score']
test_baller['Price_adjusted'] = test_baller['Price_score']
test_allrounder['Price_adjusted'] = test_allrounder['Price_score']

test_batsman_y = test_batsman['Price']
test_batsman_x = test_batsman.drop(['Price', 'Player', 'Price_score', 'Price_adjusted', 'Score', 'Team_t20', 'Team_odi'], axis = 1)
test_baller_y = test_baller['Price_score']
test_baller_x = test_baller.drop(['Price', 'Player', 'Price_score', 'Price_adjusted', 'Score', 'Team_t20', 'Team_odi', 'BBI_t20', 'BBI_odi'], axis = 1)
test_allrounder_y = test_allrounder['Price_score']
test_allrounder_x = test_allrounder.drop(['Price', 'Player', 'Price_score', 'Price_adjusted', 'Score', 'Team_t20', 'Team_odi', 'BBI_t20', 'BBI_odi'], axis = 1)

test_batsman.fillna(0,inplace=True)
test_baller.fillna(0,inplace=True)
test_allrounder.fillna(0,inplace=True)


# In[ ]:


def fix array_dim(item):
    


# In[88]:


def model_building(model, predictors, outcome, data, test_data):
    '''Function to build model, cross-validate and predict results'''
    #model.fit(data[predictors], data[outcome])  
    kf = KFold(data.shape[0], n_folds = 3)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    cv_error = np.mean(np.array(error)**2)
    #print('Cross validation Score : %s' % '{0:.3%}'.format(cv_error))
    model.fit(data[predictors],data[outcome])
    #coefficients = [model.intercept_, model.coef_]
    #print coefficients
    predictions = model.predict(test_data[predictors])
    predictions = [predictions[i][0] for i in xrange(len(predictions))]
    test_data["predicted_price"] = predictions
    #print test_data["predicted_price"]
    accuracy = np.mean(np.subtract(test_data["predicted_price"].values,test_data[outcome].values)**2)
    #print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    #test_data["Team1"] = test_data["team1"].apply(remap)
    #test_data["Team2"] = test_data["team2"].apply(remap)
    #test_data["Actual Winner"] = test_data["winner"].apply(remap)
    #test_data["Predicted Winner"] = test_data["predicted_winner"].apply(remap)
    #df = test_data[["Team1","Team2","Actual Winner", "Predicted Winner"]]
    return [accuracy, cv_error]


# In[89]:


# Models tested
model1 = lr()                         #linear regression
'''model2 = LogisticRegression()         #L2 regularization, one vs all
model3 = LogisticRegression(penalty='l1')         #L1 regularization, one vs all
model4 = LogisticRegression(solver='newton-cg', multi_class='multinomial')  #Multinomial
model5 = SVC(kernel = "linear")
model6 = DTC()
model7 = RandomForestClassifier(n_estimators=100)
'''
models = [model1]#, model2, model3, model4, model5, model6, model7]
#results = []
accuracies = []
cv_errors = []
#col = ["Intercept"] + list(train_data.columns)
#coefficients_summary = pd.DataFrame(columns= col)
#Batsman
output = ['Price_adjusted']
predictors = training_batsman.drop(['Price', 'Player', 'Price_score', 'Price_adjusted', 'Score', 'Team_t20', 'Team_odi'],1).columns
for model in models:
    [accuracy, cv_error] = model_building(model, predictors, output, training_batsman, test_batsman)
    #results.append(result)
    accuracies.append(accuracy)
    cv_errors.append(cv_error)
    #coefficients_summary = coefficients_summary.append(coefficients, ignore_index=True)
model_names = ["Linear Regression"]#, "LogisticRegression(One vs All) L2 reg", "LogisticRegression(One vs All) L1 reg", "MultinomialRegression", "SVM", "DecisionTree", "Random Forest"]
model_comparison = pd.DataFrame(columns=["Model Names", "MSE", "Cross Validation Errors"])
model_comparison["Model Names"] = model_names
model_comparison["MSE"] = accuracies
model_comparison["Cross Validation Errors"] = cv_errors


# In[90]:


model_comparison


# In[66]:


model1 = lr()
model1.fit(training_batsman[predictors],training_batsman[output])
#coefficients = [model.intercept_, model.coef_]
#print coefficients
predictions = model.predict(test_batsman[predictors])
predictions = [predictions[i][0] for i in xrange(len(predictions))]


# In[67]:


predictions

