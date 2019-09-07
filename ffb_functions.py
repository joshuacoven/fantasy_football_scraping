###################### Import Packages #############################################
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import statsmodels.formula.api as smf

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
        else: database = database.append(dfyear, ignore_index = True)

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
        if (tm_prev != tm_curr) or (tm_curr == '2TM'):
            val = 1
        else:
            val = 0
    else:
        val = np.nan
    return val

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
    database = database.drop_duplicates(subset = 'Name', keep = 'last')
    #database = database.reset_index(drop = True)
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
        else: database = database.append(dfyear, ignore_index = True)
    return database


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
    return database