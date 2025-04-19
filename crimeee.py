import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

name = ['mshafe01', 'Jmason15']

# Function to calculate year-over-year differences
def diff_data(df):
    # Split the data into two years
    df_year1 = df[df['week'] < 53].reset_index(drop=True)
    df_year2 = df[df['week'] >= 53].reset_index(drop=True)
    
    # Create a new DataFrame for differences
    diff_df = df_year2[['borough', 'week']].copy() 
    
    # Calculate year-over-year differences
    diff_df['dlcrime'] = df_year2['lcrime'].values - df_year1['lcrime'].values
    diff_df['dlpolice'] = df_year2['lpolice'].values - df_year1['lpolice'].values
    diff_df['dlun'] = df_year2['lun'].values - df_year1['lun'].values
    diff_df['dlemp'] = df_year2['lemp'].values - df_year1['lemp'].values
    diff_df['dlymale'] = df_year2['lymale'].values - df_year1['lymale'].values
    diff_df['dlwhite'] = df_year2['lwhite'].values - df_year1['lwhite'].values

    return diff_df

# Read the data
file_path = "/Users/jadenfix/Desktop/Graduate School Materials/Computing and Machine Learning/london_crime.csv"
data = pd.read_csv(file_path)

# Crime rate and police rate
data['crimerate'] = data['crime'] / data['population']
data['policerate'] = data['police'] / data['population']

# Log variables
data['lcrime'] = np.log(data['crimerate'])
data['lpolice'] = np.log(data['policerate'])
data['lemp'] = np.log(data['emp'])
data['lun'] = np.log(data['un'])
data['lymale'] = np.log(data['ymale'])
data['lwhite'] = np.log(data['white'])

# Regression model 1
model1 = smf.ols("lcrime ~ lpolice + lemp + lun + lymale + lwhite", data=data).fit()
print(model1.summary())

# Get the differences DataFrame
diff_df = diff_data(data.copy()) 

# Regression model 2
model2 = smf.ols("dlcrime ~ C(week) + dlpolice + dlemp + dlun + dlymale + dlwhite", data=diff_df).fit()
print(model2.summary())

# Regression model 3
diff_df['sixweeks'] = np.where((diff_df['week'] >= 80) & (diff_df['week'] <= 85), 1, 0)
diff_df['sixweeks_treat'] = np.where(diff_df['borough'].isin([1, 2, 3, 6, 14]) & (diff_df['sixweeks'] == 1), 1, 0)

model3 = smf.ols("dlcrime ~ C(week) + sixweeks + sixweeks_treat + dlemp + dlun + dlymale + dlwhite", data=diff_df).fit()
print(model3.summary())