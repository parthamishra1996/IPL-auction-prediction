import pandas as pd
import numpy as np

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

# Data imputation

best_batsman = batsman_2014.irow(0)
avg_batsman = batsman_2014.irow(1)
new_batsman = batsman_2014.irow(2)
best_baller = baller_2014.irow(0)
avg_baller = baller_2014.irow(1)
new_baller = baller_2014.irow(2)
best_allrounder = allrounder_2014.irow(0)
avg_allrounder = allrounder_2014.irow(1)
new_allrounder = allrounder_2014.irow(2)

def fill_empty_prices_batsman(itrm):

for x in ['14', '15', '16', '17']:
    for y in ['batsman_20', 'baller_20', 'allrounder_20']:
        typ_name = y + x
        typ = eval(typ_name)
        typ = typ.apply(fill_empty_prices, axis = 1)



# Model building

from scipy import stats

## Model1: 

## Setting training data
training_batsman = pd.concat([batsman_2014, batsman_2015, batsman_2016, odi_batsman_2014, odi_batsman_2015, odi_batsman_2016], ignore_index = True)
training_baller = pd.concat([baller_2014, baller_2015, baller_2016, odi_baller_2014, odi_baller_2015, odi_baller_2016], ignore_index = True)
training_allrounder = pd.concat([allrounder_2014, allrounder_2015, allrounder_2016, odi_allrounder_2014, odi_allrounder_2015, odi_allrounder_2016], ignore_index = True)

training_batsman = training_batsman[training_batsman['Price'] > 0].reset_index()
training_baller = training_baller[training_baller['Price'] > 0].reset_index()
training_allrounder = training_allrounder[training_allrounder['Price'] > 0].reset_index()

training_batsman_y = training_batsman['Price']
training_batsman_x = training_batsman.drop(['Price', 'Player'], axis = 1)
training_baller_y = training_baller['Price']
training_baller_x = training_baller.drop(['Price', 'Player'], axis = 1)
training_allrounder_y = training_allrounder['Price']
training_allrounder_x = training_allrounder.drop(['Price', 'Player'], axis = 1)

## Setting test data
test_batsman = batsman_2017
test_baller = baller_2017
test_allrounder = allrounder_2017

test_batsman = test_batsman[test_batsman['Price'] > 0].reset_index()
test_baller = test_baller[test_baller['Price'] > 0].reset_index()
test_allrounder = test_allrounder[test_allrounder['Price'] > 0].reset_index()

test_batsman_y = test_batsman['Price']
test_batsman_x = test_batsman.drop(['Price', 'Player'], axis = 1)
test_baller_y = test_baller['Price']
test_baller_x = test_baller.drop(['Price', 'Player'], axis = 1)
test_allrounder_y = test_allrounder['Price']
test_allrounder_x = test_allrounder.drop(['Price', 'Player'], axis = 1)

#slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)