import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#now loadinfg the data

df=pd.read_csv(r"C:\Users\rohin\Credit Risk Classifier\data\raw\cs-training.csv")

#Remove unnessary colums
df.drop(columns=['Unnamed: 0'],inplace=True)

#There are some missing data so had to handel that
df['MonthlyIncome']=df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

df['NumberOfDependents']=df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

print(df.head())

#remove unrealistic age values
df=df[df['age']>18]

#Handel Outliers
# Cap DebtRatio
upper_debt=df['DebtRatio'].quantile(0.99)

df['DebtRatio'] = df['DebtRatio'].clip(upper=upper_debt)

# Cap MonthlyIncome
upper_income = df['MonthlyIncome'].quantile(0.99)

df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=upper_income)

# VERIFY CLEANING


print(df.info())

print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:", df.shape)

print("\nCleaning Completed Successfully")

print("\nClass Distribution:")
print(df['SeriousDlqin2yrs'].value_counts())

sns.countplot(x='SeriousDlqin2yrs',data=df)
plt.title("Loan Default Distribution")
plt.show()