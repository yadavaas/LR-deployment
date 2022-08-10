import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression 

st.write("""
# Loan Status Prediction App
This app predicts the **Loan Status** of the borrower.

""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        IncomeRange = st.sidebar.selectbox('Income Range',('Not employed','$0','$1-24,999','$25,000-49,999','$50,000-74,999',
                                                          '$75,000-99,999','$100,000+'))
        Occupation = st.sidebar.selectbox('Occupation',('Professional', 'Skilled Labor', 'Executive', 'Sales - Retail',
                                                       'Laborer', 'Food Service', 'Fireman', 'Construction',
                                                       'Computer Programmer', 'Other', 'Sales - Commission',
                                                       'Retail Management', 'Engineer - Mechanical', 'Military Enlisted',
                                                       'Clerical', 'Unknown', 'Teacher', 'Clergy', 'Attorney',
                                                       'Nurse (RN)', 'Accountant/CPA', 'Analyst', 'Investor',
                                                       'Flight Attendant', 'Nurse (LPN)', 'Military Officer',
                                                       'Truck Driver', 'Administrative Assistant',
                                                       'Police Officer/Correction Officer', 'Social Worker',
                                                       'Food Service Management', 'Tradesman - Mechanic',
                                                       'Medical Technician', 'Professor', 'Postal Service',
                                                       'Waiter/Waitress', 'Civil Service', 'Pharmacist',
                                                       'Tradesman - Electrician', 'Scientist', 'Dentist',
                                                       'Engineer - Electrical', 'Architect', 'Landscaping', 'Bus Driver',
                                                       'Engineer - Chemical', 'Doctor', 'Chemist', "Teacher's Aide",
                                                       'Pilot - Private/Commercial', "Nurse's Aide", 'Religious',
                                                       'Homemaker', 'Realtor', 'Student - College Senior', 'Principal',
                                                       'Psychologist', 'Biologist', 'Tradesman - Carpenter', 'Judge',
                                                       'Car Dealer', 'Student - College Graduate Student',
                                                       'Student - College Freshman', 'Student - College Junior',
                                                       'Tradesman - Plumber', 'Student - College Sophomore',
                                                       'Student - Community College', 'Student - Technical School'))
        EmploymentStatus = st.sidebar.selectbox('Employment Status',('Employed','Full-time','Self-employed','Part-time','Retired',
                                                                  'Not employed','Other'))
        LoanOriginationQuarter = st.sidebar.selectbox('Loan Origination Quarter',('Q3 2009','Q4 2009','Q1 2010','Q2 2010','Q3 2010',
                                                                  'Q4 2010','Q1 2011', 'Q2 2011', 'Q3 2011', 'Q4 2011', 'Q1 2012',
                                                                                 'Q2 2012', 'Q3 2012','Q4 2012', 'Q1 2013', 'Q2 2013',
                                                                                 'Q3 2013', 'Q4 2013', 'Q1 2014'))
        ProsperScore = st.sidebar.slider('Prosper Score', 1, 11, 6)
        BorrowerRate = st.sidebar.slider('Borrower Rate', 0.04, 0.36, 0.19)
        BorrowerAPR = st.sidebar.slider('Borrower APR', 0.04, 0.423, 0.22)
        DebtToIncomeRatio = st.sidebar.slider('Debt To Income Ratio', 0.0, 10.0, 0.26)
        BankcardUtilization = st.sidebar.slider('BankcardUtilization', 0.0, 2.5, 0.56)
        Investors = st.sidebar.slider('Investors', 1, 1189, 1)
        Term = st.sidebar.slider('Term', 12,60,36)
        DelinquenciesLast7Years = st.sidebar.slider('DelinquenciesLast7Years', 0,99,0)
        MonthlyLoanPayment = st.sidebar.slider('MonthlyLoanPayment', 0.0, 2251.5, 291.0)
        LoanOriginalAmount = st.sidebar.slider('LoanOriginalAmount', 1000,35000,9000)
        ProsperRatingNumeric = st.sidebar.slider('ProsperRatingNumeric', 1,7,4)
        LenderYield = st.sidebar.slider('LenderYield', 0.0,0.36,0.18)
        EstimatedEffectiveYield = st.sidebar.slider('EstimatedEffectiveYield', -0.18,0.32, 0.17)
        EstimatedLoss = st.sidebar.slider('EstimatedLoss', 0.0, 0.37,0.08)
        EstimatedReturn = st.sidebar.slider('EstimatedReturn', -0.18,0.3,0.1)
        EmploymentStatusDuration = st.sidebar.slider('EmploymentStatusDuration', 0,755,100)
        TotalCreditLinespast7years = st.sidebar.slider('TotalCreditLinespast7years', 2,125,27)
        OpenRevolvingMonthlyPayment = st.sidebar.slider('OpenRevolvingMonthlyPayment', 0,1180,430)
        RevolvingCreditBalance = st.sidebar.slider('RevolvingCreditBalance', 0,45000,17940)
        AvailableBankcardCredit = st.sidebar.slider('AvailableBankcardCredit', 0,33000,11400)
        TotalTrades = st.sidebar.slider('TotalTrades', 1,122,24)
        StatedMonthlyIncome = st.sidebar.slider('StatedMonthlyIncome', 0,15000,6000)
        LoanMonthsSinceOrigination = st.sidebar.slider('LoanMonthsSinceOrigination', 0,56,16)
      
        data = {'IncomeRange': IncomeRange,
                'ProsperScore': ProsperScore,
                'BorrowerRate': BorrowerRate,
                'BorrowerAPR': BorrowerAPR,
                'Occupation': Occupation,
                'DebtToIncomeRatio': DebtToIncomeRatio,
                'BankcardUtilization': BankcardUtilization,
                'Investors': Investors,
                'Term': Term,
                'DelinquenciesLast7Years': DelinquenciesLast7Years,
                'MonthlyLoanPayment': MonthlyLoanPayment,
                'LoanOriginalAmount': LoanOriginalAmount,
                'ProsperRating (numeric)': ProsperRatingNumeric,
                'LenderYield': LenderYield,
                'EstimatedEffectiveYield': EstimatedEffectiveYield,
                'EstimatedLoss': EstimatedLoss,
                'EstimatedReturn': EstimatedReturn,
                'EmploymentStatus': EmploymentStatus,
                'EmploymentStatusDuration': EmploymentStatusDuration,
                'TotalCreditLinespast7years': TotalCreditLinespast7years,
                'OpenRevolvingMonthlyPayment': OpenRevolvingMonthlyPayment,
                'RevolvingCreditBalance': RevolvingCreditBalance,
                'AvailableBankcardCredit': AvailableBankcardCredit,
                'TotalTrades': TotalTrades,
                'StatedMonthlyIncome': StatedMonthlyIncome,
                'LoanMonthsSinceOrigination': LoanMonthsSinceOrigination,
                'LoanOriginationQuarter': LoanOriginationQuarter}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
data = pd.read_csv('updated_data.csv')
df_selected = data[['IncomeRange','ProsperScore', 'BorrowerRate', 'BorrowerAPR', 'Occupation', 
           'DebtToIncomeRatio', 'BankcardUtilization','Investors','Term', 'DelinquenciesLast7Years',
           'MonthlyLoanPayment', 'LoanOriginalAmount', 'ProsperRating (numeric)', 'LenderYield', 'EstimatedEffectiveYield',
           'EstimatedLoss', 'EstimatedReturn', 'EmploymentStatus', 'EmploymentStatusDuration', 'TotalCreditLinespast7years',
           'OpenRevolvingMonthlyPayment', 'RevolvingCreditBalance', 'AvailableBankcardCredit', 'TotalTrades',
           'StatedMonthlyIncome', 'LoanMonthsSinceOrigination', 'LoanOriginationQuarter', 
           'LoanStatus' ]]
df_new = df_selected.drop(columns=['LoanStatus'])
df = pd.concat([input_df,df_new],axis=0)

# Encoding of ordinal features
#Importing the LabelEncoder module
from sklearn.preprocessing import LabelEncoder
#Lebel Encoding the variables
LE = LabelEncoder()
df['IncomeRange'] = LE.fit_transform(df['IncomeRange'])
df['Occupation'] = LE.fit_transform(df['Occupation'])
df['EmploymentStatus'] = LE.fit_transform(df['EmploymentStatus'])
df['LoanOriginationQuarter'] = LE.fit_transform(df['LoanOriginationQuarter'])

#applying feature engineering part 

#storing variable df into variable X
X = df







df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
LR = pickle.load(open('LR_pickle.pkl', 'rb'))

# Apply model to make predictions
prediction = LR.predict(df)
prediction_proba = LR.predict_proba(df)


st.subheader('Prediction')
Loan_Status = np.array(['High Risk','Accepted'])
st.write(Loan_Status[prediction])

st.subheader('Prediction Probability')
st.write('0 = High Risk | 1 = Accepted')
st.write(prediction_proba)