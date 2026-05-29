import pandas as pd

#load cleaned dataset
df=pd.read_csv(r"C:\Users\rohin\Credit Risk Classifier\data\raw\cs-training.csv")

#remove unnessary column
df.drop(columns=['Unnamed: 0'],inplace=True)

#Fill missing values
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

#remove invalid ages
df=df[df['age']>18]

#-------------
#Feature Engineering
#-------------

#Income Per dependent
df['IncomePerDependent']=(df['MonthlyIncome']/(df['NumberOfDependents']+1))

#Debt to income ratio
df['DebtToIncome']=(df['DebtRatio']/(df['MonthlyIncome']+1))

#Late payment Flag
df['LatePaymentFlag']=(df['NumberOfTimes90DaysLate']>0).astype(int)

#Total past due count
df['TotalPastDueCounts']=(df['NumberOfTime30-59DaysPastDueNotWorse'] +
    df['NumberOfTime60-89DaysPastDueNotWorse'] +
    df['NumberOfTimes90DaysLate'])


###### Display new features ######

print(df.head())

print("New features added sucessfully")
print("Columns:")

print(df.columns)