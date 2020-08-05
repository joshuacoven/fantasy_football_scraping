###################### Import Packages #############################################
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import statsmodels.formula.api as smf
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skgarden import RandomForestQuantileRegressor
import matplotlib.pyplot as plt

####################################### Functions ######################################

## assemble NFL data from pro football focus
def data_assembly(start_year, current_year):
    database = []
    for x in range(start_year, current_year):
        page = requests.get("https://www.pro-football-reference.com/years/%d/fantasy.htm" % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 1)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: database = database.append(dfyear, ignore_index = True, sort = False)

    #Rename columns, eliminate duplicate column titles as rows
    database = database.rename(columns = {'Player':'Name', 'Att':'PaAtt', 'Yds':'PaYds', 'TD':'PaTD','Att.1':'RuAtt', 'Yds.1':'RuYds', 'TD.1':'RuTD', 'Y/A':'RuY/A', 'Y/R':'ReYds/R', 'Att.2':'ReAtt', 'Yds.2':'ReYds', 'TD.2':'ReTD'})
    database = database[database.Rk != 'Rk']
    
    # clean up artifacts at the end of names
    database['Name'] = database['Name'].apply(lambda x: x[0:len(x)-1] if x[len(x)-1] == '+' else x)
    database['Name'] = database['Name'].apply(lambda x: x[0:len(x)-1] if x[len(x)-1] == '*' else x)
    return database


#Pull rookie data from pro football focus
def rookie_assembly(start_year, current_year):
    database = []
    for x in range(start_year, current_year):
        page = requests.get("https://www.pro-football-reference.com/years/%d/draft.htm" % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 1)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: database = database.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database = database.rename(columns = {'Player':'Name', 'Att':'PaAtt', 'Yds':'PaYds', 'TD':'PaTD','Att.1':'RuAtt', 'Yds.1':'RuYds', 'TD.1':'RuTD', 'Y/A':'RuY/A', 'Y/R':'ReYds/R', 'Att.2':'ReAtt', 'Yds.2':'ReYds', 'TD.2':'ReTD', 'Pos':'FantPos', 'College/Univ':'College'})
    database = database[database.Rnd != 'Rnd']
    database = database[['Pick', 'Tm', 'Name', 'Age', 'College', 'Year', 'FantPos']]
    # keep fantasy relevant positions
    database = database.loc[(database.FantPos == 'QB') 
             | (database.FantPos == 'WR')
            | (database.FantPos == 'RB')
            | (database.FantPos == 'TE')].reset_index(drop = True)
    return database


# pull combine data from pro football focus
def combine_assembly(start_year, current_year):
    database = []
    for x in range(start_year, current_year):
        page = requests.get('https://www.pro-football-reference.com/draft/%d-combine.htm' % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 0)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: database = database.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database = database.rename(columns = {'Player':'Name', 'Broad Jump':'Broad_Jump', '3Cone':'Three_Cone', '40yd':'Dash'})
    database = database[database.Name != 'Player']
    database = database.rename(columns = {'Pos':'FantPos'}) # for merge
    database = database[['Name', 'Year', 'School', 'FantPos', 'Ht', 'Wt', 'Dash', 'Vertical', 'Bench', 'Broad_Jump', 'Three_Cone', 'Shuttle']]
    #keep fantasy relevant positions
    database = database.loc[(database.FantPos == 'QB') 
             | (database.FantPos == 'WR')
            | (database.FantPos == 'RB')
            | (database.FantPos == 'TE')].reset_index(drop = True) 

    database['height'] = database['Ht'].apply(lambda x: 12 * float(x.split('-')[0]) + float(x.split('-')[1]) \
                                              if isinstance(x, str) else np.nan)

    # impute combine statistics based on position mean
    #cols = ['height', 'Wt', 'Dash', 'Vertical', 'Bench', 'Broad_Jump', 'Three_Cone', 'Shuttle']
    #for i in cols:
     #   database[i] = database[i].astype(float)
      #  database[i] = database[i].fillna(database.groupby('FantPos')[i].transform('mean'))
    return database


#Pull college stats from college football focus
def college_assembly(start_year, current_year):
    database_qb = []
    for x in range(start_year, current_year):
        page = requests.get('https://www.sports-reference.com/cfb/years/%d-passing.html' % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 1)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database_qb = dfyear
        else: database_qb = database_qb.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database_qb = database_qb.rename(columns = {'Player':'Name', 'Att':'PaAtt', 'Yds':'PaYds', 'TD':'PaTD', 'TD.1':'RuTD', 'Yds.1':'RuYds', 'Att.1':'RuAtt'})
    database_qb = database_qb[database_qb.Rk != 'Rk']
    database_qb['Name'] = database_qb['Name'].apply(lambda x: x[0:len(x)-1] if x[len(x)-1] == '*' else x)
    
    database_rb = []
    for x in range(start_year, current_year):
        page = requests.get('https://www.sports-reference.com/cfb/years/%d-rushing.html' % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 1)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database_rb = dfyear
        else: database_rb = database_rb.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database_rb = database_rb.rename(columns = {'Player':'Name', 'Att':'RuAtt', 'TD':'RuTD', 'Yds':'RuYds', 'Yds.1':'ReYds', 'TD.1':'ReTD'})
    database_rb = database_rb[database_rb.Rk != 'Rk']
    database_rb['Name'] = database_rb['Name'].apply(lambda x: x[0:len(x)-1] if x[len(x)-1] == '*' else x)
        
    database_wr = []
    for x in range(start_year, current_year):
        page = requests.get('https://www.sports-reference.com/cfb/years/%d-receiving.html' % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 1)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database_wr = dfyear
        else: database_wr = database_wr.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database_wr = database_wr.rename(columns = {'Player':'Name', 'Yds':'ReYds', 'Yds.1':'RuYds', 'TD':'ReTD', 'TD.1':'RuTD', 'Att':'RuAtt'})
    database_wr = database_wr[database_wr.Rk != 'Rk']
    database_wr['Name'] = database_wr['Name'].apply(lambda x: x[0:len(x)-1] if x[len(x)-1] == '*' else x)
    
    database = database_qb.append(database_wr, ignore_index = True, sort = False)
    database = database.append(database_rb, ignore_index = True, sort = False)
    database = database[['Name', 'Year', 'School', 'Conf', 'G', 'Cmp', 'PaAtt', 'Pct', 'PaYds', 'PaTD', 'Int', 'RuYds', 'RuTD', 'RuAtt'
        , 'ReYds', 'ReTD', 'Rec']]
    database = database.drop_duplicates(subset = ['Name', 'School', 'Year'], keep = 'first')
    database['latest_year'] = database.groupby('Name')['Year'].rank('dense', ascending = False)
    database = database.loc[database.latest_year == 1].reset_index(drop = True).drop('latest_year', axis = 1)
    database = database.fillna(0) # this fills in stats that are missing, works
    database['Name'] = database['Name'].str.replace('Joshua Jacobs', 'Josh Jacobs') # make names match across databases
    return database


def ADP_assembly(start_year, current_year):
    database = []
    for x in range(start_year, current_year):
        page = requests.get("https://fantasyfootballcalculator.com/adp/ppr/12-team/all/%d" % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 0)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: 
            database = database.append(dfyear, ignore_index = True)
        database = database[['Name', 'Year', 'Std.Dev']]
        # the year is off of the profootball focus data by 1
        database = database.replace('Odell Beckham Jr', 'Odell Beckham')
    return database

## alternate adp datasource, better data stability, many more years
def new_assembly(start_year, current_year, rk_opt):
    database = []
    for x in range(start_year, current_year):
        page = requests.get("http://www03.myfantasyleague.com/%d/adp?COUNT=500&POS=*&ROOKIES=0&INJURED=0&CUTOFF=5&FRANCHISES=-1&IS_PPR=1&IS_KEEPER=0&IS_MOCK=0&TIME=" % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find_all('table')[1]
        #print(soup.find_all('table'))
        df = pd.read_html(str(table), header = 0)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: 
            database = database.append(dfyear, ignore_index = True)
        test = database
        new_test = test.loc[test.Player.str.contains('QB')
            | test.Player.str.contains('WR')
            | test.Player.str.contains('RB')
            | test.Player.str.contains('TE')].reset_index(drop = True)
        new_test['List'] = new_test.Player.str.replace('*', '').str.replace(',', '').str.split(' ')
        #new_test['FantPos'] = new_test['List'].apply(lambda x: x[len(x)-1])
        new_test['Tm'] = new_test['List'].apply(lambda x: x[len(x)-2])
        new_test['Name'] = new_test['List'].apply(lambda x: x[len(x)-3] + ' ' + x[0])
        new_test['Name'] = new_test['Name']  + new_test['List'].apply(lambda x: ' ' + x[1] if len(x) == 5 else '')
        new_test = new_test[['Name', 'Tm', 'Avg. Pick', 'Year']]  
        new_test['Tm'] = new_test['Tm'].str.replace('JAC', 'JAX')
        new_test['Tm'] = new_test['Tm'].str.replace('KCC', 'KAN')
        new_test['Tm'] = new_test['Tm'].str.replace('GBP', 'GNB')
        new_test['Tm'] = new_test['Tm'].str.replace('NOS', 'NOR')        
    return new_test.rename(columns = {'Avg. Pick': 'Overall'})


tm_dict = {'Pittsburgh': 'PIT',
 'Philadelphia': 'PHI',
 'New England': 'NWE',
 'Minnesota': 'MIN',
 'Carolina': 'CAR',
 'LA Rams': 'LAR',
 'New Orleans': 'NOR',
 'Jacksonville': 'JAX',
 'Kansas City': 'KAN',
 'Atlanta': 'ATL',
 'LA Chargers': 'LAC',
 'Seattle': 'SEA',
 'Buffalo': 'BUG',
 'Dallas': 'DAL',
 'Tennessee': 'TEN',
 'Detroit': 'DET',
 'Baltimore': 'BAL',
 'Arizona': 'ARI',
 'Washington': 'WAS',
 'Green Bay': 'GNB',
 'Cincinnati': 'CIN',
 'Oakland': 'OAK',
 'San Francisco': 'SFO',
 'Miami': 'MIA',
 'Denver': 'DEN',
 'NY Jets': 'NYJ',
 'Tampa Bay': 'TAM',
 'Chicago': 'CHI',
 'Indianapolis': 'IND',
 'Houston': 'HOU',
 'NY Giants': 'NYG',
 'Cleveland': 'CLE'}

def team_assembly(start_year, current_year):
    database = []
    for x in range(start_year, current_year):
        page = requests.get('https://www.teamrankings.com/nfl/trends/win_trends/?sc=is_regular_season&range=yearly_%d' % x)
        soup = BeautifulSoup(page.content, 'html.parser')
        #print(soup.prettify)
        #print(soup.find_all('table'))
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table), header = 0)
        dfyear = df[0]
        dfyear['Year'] = x
        if x == start_year:
            database = dfyear
        else: database = database.append(dfyear, ignore_index = True)

    #Rename columns, eliminate duplicate column titles as rows
    database = database.rename(columns = {'Win %':'Win_PCT'})
    database['Tm'] = database['Team'].apply(lambda x: tm_dict[x])
    database = database[['Tm', 'Win_PCT', 'Year']]
    database['Win_PCT'] = database['Win_PCT'].apply(lambda x: float(x[0:len(x)-1]))
    return database




# For shifting columns to prepare across years
def shift_col(frame, new_name, col_to_shift, magnitude):
    frame1 = frame.sort_values(['Name', 'Year']).reset_index(drop = True)
    frame1[new_name] = frame1[col_to_shift].shift(magnitude)
    for i in range(len(frame1) - 1):
        if frame1.loc[i, 'Name'] != frame1.loc[i+1, 'Name']:
            if magnitude == -1:
                frame1.loc[i, new_name] = np.nan
            elif magnitude == 1:
                frame1.loc[i+1, new_name] = np.nan
    return frame1

# To make a flag for switching teams
def new_team(tm_prev, tm_curr):
    if pd.notna(tm_prev) and pd.notna(tm_curr):
        if (tm_prev != tm_curr) or (tm_curr == '2TM') or (tm_prev == '2TM'):
            val = 1
        else:
            val = 0
    else:
        val = np.nan
    return val

### random forest model #################################################
# inputs, a dictionary where each position has a dataframe of desired features, no nans, imputed values
# a year one year before the year you want to predict for, ex: 2018 trains the model to make predictions about 2019
# a window size, number of years of data you want to include in the model
def rf_model(pos_dict, predict_year, sz):
    model_dict = {}
    predict_dict = {}
    outcomes = {}
    for pos in pos_dict:
        print(pos)
        target = copy.deepcopy(pos_dict[pos])
        
        # team dummy variables
        dum = pd.get_dummies(target.Tm)
        target = target.drop('Tm', axis = 1)
        target = pd.concat([target, dum], axis=1)
        
        # save these values to evaluate predictions later
        outcomes[pos] = target.loc[target.Year == predict_year]\
            .reset_index(drop = True)[['Name', 'pts_next_year']]
        
        # set aside data to use for model prediction when the model is done
        predict_dict[pos] = target.loc[target.Year == predict_year]\
            .reset_index(drop = True)\
            .drop(['Year', 'pts_next_year', 'Name'], axis=1)
        
        # make sure new values arent used in the modeling
        target = target.loc[target.Year < predict_year].reset_index(drop = True)
        # only use 'sz' years of data before prediction year
        target = target.loc[target.Year > predict_year - sz].drop(['Year', 'Name'], axis=1)

        # separate labels, targets, features
        labels = np.array(target['pts_next_year'])
        target = target.drop(['pts_next_year'], axis = 1)
        features = np.array(target)
        feature_list = list(target.columns)
        
        # run model 
        rf = RandomForestRegressor(n_estimators = 3000, random_state = 42)
        rf.fit(features, labels)
        model_dict[pos] = rf

        # Get numerical feature importances
        importances = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    ## get ouptuts
    final_dict = copy.deepcopy(predict_dict)
    for pos in predict_dict:
        model = model_dict[pos]
        final_dict[pos]['prediction'] = model.predict(predict_dict[pos])
        final_dict[pos]['Names'] = outcomes[pos]['Name']
        final_dict[pos]['pts_next_year'] = outcomes[pos]['pts_next_year']
    return final_dict


### Quantile random forest model #################################################
# inputs, a dictionary where each position has a dataframe of desired features, no nans, imputed values
# a year one year before the year you want to predict for, ex: 2018 trains the model to make predictions about 2019
# a window size, number of years of data you want to include in the model
def rfqr_model(pos_dict, predict_year, sz):
    model_dict = {}
    predict_dict = {}
    outcomes = {}
    for pos in pos_dict:
        print(pos)
        target = copy.deepcopy(pos_dict[pos])
        
        # team dummy variables
        dum = pd.get_dummies(target.Tm)
        target = target.drop('Tm', axis = 1)
        target = pd.concat([target, dum], axis=1)
        
        # save these values to evaluate predictions later
        outcomes[pos] = target.loc[target.Year == predict_year]\
            .reset_index(drop = True)[['Name', 'pts_next_year']]
        
        # set aside data to use for model prediction when the model is done
        predict_dict[pos] = target.loc[target.Year == predict_year]\
            .reset_index(drop = True)\
            .drop(['Year', 'pts_next_year', 'Name'], axis=1)
        
        # make sure new values arent used in the modeling
        target = target.loc[target.Year < predict_year].reset_index(drop = True)
        # only use 'sz' years of data before prediction year
        target = target.loc[target.Year > predict_year - sz].drop(['Year', 'Name'], axis=1)

        # separate labels, targets, features
        labels = np.array(target['pts_next_year'])
        target = target.drop(['pts_next_year'], axis = 1)
        features = np.array(target)
        feature_list = list(target.columns)
        
        # run model 
        rfqr = RandomForestQuantileRegressor(random_state=0, n_estimators=3000)
        rfqr.fit(features, labels)
        model_dict[pos] = rfqr

        
        
        
        
        #
        upper = np.concatenate(([], rfqr.predict(predict_dict[pos], quantile=98.5)))
        lower = np.concatenate(([], rfqr.predict(predict_dict[pos], quantile=2.5)))
        median = np.concatenate(([], rfqr.predict(predict_dict[pos], quantile=50)))

        #interval = upper - lower
        #sort_ind = np.argsort(interval)
        y_true_all = outcomes[pos]['pts_next_year']#[sort_ind]
        upper = upper#[sort_ind]
        lower = lower#[sort_ind]
        median = median#[sort_ind]
        #mean = (upper + lower) / 2

        # Center such that the mean of the prediction interval is at 0.0
        #y_true_all -= mean
        #upper -= mean
        #lower -= mean

        plt.plot(y_true_all, "ro")
        plt.fill_between(
            np.arange(len(upper)), lower, upper, alpha=0.2, color="r",
            label="Pred. interval")
        plt.plot(median)
        plt.xlabel("X variable")
        plt.ylabel("Points")
        plt.xlim([0, 100])
        plt.show()
        
        
        
        
        
        # Get numerical feature importances
        importances = list(rfqr.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    ## get ouptuts
    final_dict = copy.deepcopy(predict_dict)
    for pos in predict_dict:
        model = model_dict[pos]
        final_dict[pos]['prediction'] = model.predict(predict_dict[pos])
        final_dict[pos]['Names'] = outcomes[pos]['Name']
        final_dict[pos]['pts_next_year'] = outcomes[pos]['pts_next_year']
    return final_dict


