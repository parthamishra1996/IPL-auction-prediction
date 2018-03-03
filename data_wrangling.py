import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Auction Prices
auction_2018 = pd.read_csv("Data/Auction_2018.csv")
auction_2017 = pd.read_csv("Data/Auction_2017.csv")
auction_2016 = pd.read_csv("Data/Auction_2016.csv")
auction_2015 = pd.read_csv("Data/Auction_2015.csv")

# T20I stats
batsman_2017 = pd.read_csv("Data/batsman_2017.csv")
batsman_2016 = pd.read_csv("Data/batsman_2016.csv")
batsman_2015 = pd.read_csv("Data/batsman_2015.csv")

baller_2017 = pd.read_csv("Data/baller_2017.csv")
baller_2016 = pd.read_csv("Data/baller_2016.csv")
baller_2015 = pd.read_csv("Data/baller_2015.csv")

allrounder_2017 = pd.read_csv("Data/allrounder_2017.csv")
allrounder_2016 = pd.read_csv("Data/allrounder_2016.csv")
allrounder_2015 = pd.read_csv("Data/allrounder_2015.csv")
# IPL data

ipl = pd.read_csv("Data/deliveries.csv")
#print auction_2018.head()

# Data preprocessing

def fix_name_bracket(txt):
    txt = re.sub("[\(].*?[\)]","",txt)
    return txt[:-1]

def standardize_name(txt):
    txt = txt.split(' ')
    initials = ''
    for x in xrange(len(txt) - 1):
        if txt[x][0].isupper():
            initials += txt[x][0]
            if (len(txt[x]) > 1) and (txt[x][1].isupper()):
                initials += txt[x][1]    
    return ' '.join([initials, txt[-1]])
    
def fix_name_2016(txt):
    c = txt.split(',')
    d = c[1].split(' ')
    return " ".join([d[1][:len(d[1])/2], d[-1]])
def comma_remove(item):
    return item.replace(',','')
def star_remove(item):
    return item.replace('*','')

for x in ['15', '16', '17']:
    for y in ['batsman_20', 'baller_20', 'allrounder_20']:
        z = eval(y + x)
        z['Player'] = z['Player'].apply(fix_name_bracket)

auction_2015.drop(auction_2015.columns[[1,2,3,4,5,6]], axis =1, inplace = True)
auction_2015['Player'] = auction_2015['Name']
auction_2015.drop(['Name'], axis=1, inplace = True)
auction_2015[['Team', 'Price']] = auction_2015[auction_2015.columns[[0,1]]]
auction_2015.drop(auction_2015.columns[[0,1]], axis =1, inplace = True)

auction_2016['Player'] = auction_2016['Player'].apply(fix_name_2016)
auction_2016['Price'] = auction_2016[auction_2016.columns[-1]]
auction_2016.drop(auction_2016.columns[[0,3,4]], axis=1, inplace=True)

auction_2017['Price'] = auction_2017[auction_2017.columns[2]]
auction_2017.drop(auction_2017.columns[2], axis=1, inplace=True)
auction_2017['Price'] = auction_2017['Price'].apply(comma_remove)
auction_2017['Price'] = auction_2017['Price'].astype(int)/100000
auction_2017['Player'] = auction_2017['Player'].apply(star_remove)

auction_2018['Price'] = auction_2018[auction_2018.columns[2]]
auction_2018.drop(auction_2018.columns[2], axis=1, inplace=True)
auction_2018['Price'] = auction_2018['Price'].apply(comma_remove)
auction_2018['Price'] = auction_2018['Price'].astype(int)/100000
auction_2018['Player'] = auction_2018['Player'].apply(star_remove)

## Converted names in both auction and stats files to same format
for x in ['15', '16', '17', '18']:
    z = eval('auction_20' + x)
    z['Player'] = z['Player'].apply(standardize_name)

## Merging price data with player stats
for x in ['15', '16', '17']:
    auc_name = 'auction_20' + str(int(x) + 1)# Prices match with next year of stat
    auc = eval(auc_name)
    
    for y in ['batsman_20', 'baller_20', 'allrounder_20']:        
        typ_name = y + x
        typ = eval(typ_name)
        #typ_name_ = y[:-3]
        #typ_ = eval(typ_name_)
        team = []
        price = []
        players = typ['Player'].values
        for p in players:
            pl = auc[auc['Player'] == p]
            if len(pl) > 0:
                team.append(pl['Team'].values[0])
                price.append(pl['Price'].values[0])
            else:
                team.append(np.nan)
                price.append(np.nan)
        typ['Team'] = team
        typ['Price'] = price

for x in ['15', '16', '17']:
    for y in ['batsman_20', 'baller_20', 'allrounder_20']:
        typ_name = y + x
        typ = eval(typ_name)
        typ.replace('-', '0', inplace = True)
        
        if y == 'batsman_20':
            typ['HS'] = typ['HS'].apply(star_remove)
            typ['Inns'] = typ['Inns'].astype('int')
            typ['NO'] = typ['NO'].astype('int')
            typ['Runs'] = typ['RunsDescending'].astype('int')
            typ['Ave'] = typ['Ave'].astype('float')
            typ['SR'] = typ['SR'].astype('float')
            typ['100s'] = typ['100'].astype('int')
            typ['50s'] = typ['50'].astype('int')
            typ['0s'] = typ['0'].astype('int')
            typ['4s'] = typ['4s'].astype('int')
            typ['6s'] = typ['6s'].astype('int')
            typ.drop(['BF','100','50','0','RunsDescending'], axis=1, inplace=True)
        elif y == 'baller_20':
            typ['Inns'] = typ['Inns'].astype('int')
            typ['Overs'] = typ['Overs'].astype('float')
            typ['Mdns'] = typ['Mdns'].astype('int')
            typ['Runs'] = typ['Runs'].astype('int')
            typ['Wkts'] = typ['WktsDescending'].astype('int')
            #typ['BBI'] = typ['BBI'].astype('int')
            typ['Ave'] = typ['Ave'].astype('float')
            typ['Econ'] = typ['Econ'].astype('float')
            typ['SR'] = typ['SR'].astype('float')
            typ['4s'] = typ['4'].astype('int')
            typ['5s'] = typ['5'].astype('int')
            typ.drop(['WktsDescending','4','5'], axis=1, inplace=True)
        elif y == 'allrounder_20':
            typ['HS'] = typ['HS'].apply(star_remove)
            typ['Matches'] = typ['MatDescending']
            typ['Runs'] = typ['Runs'].astype('int')
            typ['HS'] = typ['HS'].astype('int')
            typ['Bat Av'] = typ['Bat Av'].astype('float')
            typ['100s'] = typ['100'].astype('int')
            typ['Wkts'] = typ['Wkts'].astype('int')
            #typ['BBI'] = typ['BBI'].astype('int')
            typ['Bowl Av'] = typ['Bowl Av'].astype('float')
            typ['5s'] = typ['5'].astype('int')
            typ['Ave Diff'] = typ['Ave Diff'].astype('float')
            typ.drop(['MatDescending','100','5'], axis=1, inplace=True)         
print batsman_2016.head()
print batsman_2016.dtypes
print baller_2016.head()
print baller_2016.dtypes
print allrounder_2016.head()
print allrounder_2016.dtypes
# Remove * and check for data types in final dataframe
# Merge for last 3 years to obtain training dataset, 
# test dataset is 2017-18