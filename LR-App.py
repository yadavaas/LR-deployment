import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.write("""
## Team - A
""")
html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Status Prediction App</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.write("""

This app predicts the **Loan Status** of the borrower.
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        EmploymentStatus = st.sidebar.selectbox('Employment Status', ('Employed', 'Full-time', 'Not employed',
                                                                      'Part-time', 'Retired', 'Self-employed',
                                                                      'Other'))
        IncomeRange = st.sidebar.selectbox('Income Range', ('$0', '$1-24,999', '$25,000-49,999', '$50,000-74,999',
                                                            '$75,000-99,999', '$100,000+', 'Not employed'))
        LoanOriginationQuarter = st.sidebar.selectbox('Loan Origination Quarter', ('Q3 2009', 'Q4 2009', 'Q1 2010',
                                                                                   'Q2 2010', 'Q3 2010', 'Q4 2010',
                                                                                   'Q1 2011', 'Q2 2011', 'Q3 2011',
                                                                                   'Q4 2011', 'Q1 2012', 'Q2 2012',
                                                                                   'Q3 2012', 'Q4 2012', 'Q1 2013',
                                                                                   'Q2 2013', 'Q3 2013', 'Q4 2013',
                                                                                   'Q1 2014'))
        Occupation = st.sidebar.selectbox('Occupation', ('Accountant/CPA', 'Administrative Assistant', 'Analyst',
                                                         'Architect', 'Attorney', 'Biologist', 'Bus Driver',
                                                         'Car Dealer', 'Chemist', 'Civil Service', 'Clergy', 'Clerical',
                                                         'Computer Programmer', 'Construction', 'Dentist', 'Doctor',
                                                         'Engineer - Chemical', 'Engineer - Electrical',
                                                         'Engineer - Mechanical' 'Executive', 'Fireman',
                                                         'Flight Attendant', 'Food Service', 'Food Service Management',
                                                         'Homemaker', 'Investor', 'Judge', 'Laborer', 'Landscaping',
                                                         'Medical Technician', 'Military Enlisted', 'Military Officer',
                                                         'Nurse (LPN)', 'Nurse (RN)', "Nurse's Aide", 'Other',
                                                         'Pharmacist', 'Pilot - Private/Commercial',
                                                         'Police Officer/Correction Officer', 'Postal Service',
                                                         'Principal', 'Professional', 'Professor', 'Psychologist',
                                                         'Realtor', 'Religious', 'Retail Management',
                                                         'Sales - Commission', 'Sales - Retail', 'Scientist',
                                                         'Skilled Labor', 'Social Worker', 'Student - College Freshman',
                                                         'Student - College Graduate Student', 'Student - College Junior',
                                                         'Student - College Senior', 'Student - College Sophomore',
                                                         'Student - Community College', 'Student - Technical School',
                                                         'Teacher', "Teacher's Aide", 'Tradesman - Carpenter',
                                                         'Tradesman - Electrician', 'Tradesman - Mechanic',
                                                         'Tradesman - Plumber', 'Truck Driver', 'Unknown',
                                                         'Waiter/Waitress'))
        AvailableBankcardCredit = st.sidebar.slider('Available Bank card Credit', 0, 33000, 11400)
        BankcardUtilization = st.sidebar.slider('Bank card Utilization', 0.0, 2.5, 0.56)
        BorrowerAPR = st.sidebar.slider('Borrower APR', 0.04, 0.423, 0.22)
        BorrowerRate = st.sidebar.slider('Borrower Rate', 0.04, 0.36, 0.19)
        DebtToIncomeRatio = st.sidebar.slider('Debt To Income Ratio', 0.0, 10.0, 0.26)
        DelinquenciesLast7Years = st.sidebar.slider('Delinquencies Last 7 Years', 0, 99, 0)
        EmploymentStatusDuration = st.sidebar.slider('Employment Status Duration', 0, 755, 100)
        EstimatedEffectiveYield = st.sidebar.slider('Estimated Effective Yield', -0.18, 0.32, 0.17)
        EstimatedLoss = st.sidebar.slider('Estimated Loss', 0.0, 0.37, 0.08)
        EstimatedReturn = EstimatedEffectiveYield - EstimatedLoss
        Investors = st.sidebar.slider('Investors', 1, 1189, 1)
        LenderYield = st.sidebar.slider('Lender Yield', 0.0, 0.36, 0.18)
        LoanOriginalAmount = st.sidebar.slider('Loan Original Amount', 1000, 35000, 9000)
        OpenRevolvingMonthlyPayment = st.sidebar.slider('Open Revolving Monthly Payment', 0, 1180, 430)
        ProsperRatingNumeric = st.sidebar.slider('Prosper Rating (Numeric)', 1, 7, 4)
        ProsperScore = st.sidebar.slider('Prosper Score', 1, 11, 6)
        RevolvingCreditBalance = st.sidebar.slider('Revolving Credit Balance', 0, 45000, 17940)
        StatedMonthlyIncome = st.sidebar.slider('Stated Monthly Income', 0, 15000, 6000)
        Term = st.sidebar.slider('Term', 12, 60, 36)
        TotalCreditLinespast7years = st.sidebar.slider('Total Credit Lines past 7 years', 2, 125, 27)
        TotalTrades = st.sidebar.slider('Total Trades', 1, 122, 24)

        data = {'AvailableBankcardCredit': AvailableBankcardCredit,
                'BankcardUtilization': BankcardUtilization,
                'BorrowerAPR': BorrowerAPR,
                'BorrowerRate': BorrowerRate,
                'DebtToIncomeRatio': DebtToIncomeRatio,
                'DelinquenciesLast7Years': DelinquenciesLast7Years,
                'EmploymentStatus': EmploymentStatus,
                'EmploymentStatusDuration': EmploymentStatusDuration,
                'EstimatedEffectiveYield': EstimatedEffectiveYield,
                'EstimatedLoss': EstimatedLoss,
                'EstimatedReturn': EstimatedReturn,
                'IncomeRange': IncomeRange,
                'Investors': Investors,
                'LenderYield': LenderYield,
                'LoanOriginalAmount': LoanOriginalAmount,
                'LoanOriginationQuarter': LoanOriginationQuarter,
                'Occupation': Occupation,
                'OpenRevolvingMonthlyPayment': OpenRevolvingMonthlyPayment,
                'ProsperRating (numeric)': ProsperRatingNumeric,
                'ProsperScore': ProsperScore,
                'RevolvingCreditBalance': RevolvingCreditBalance,
                'StatedMonthlyIncome': StatedMonthlyIncome,
                'Term': Term,
                'TotalCreditLinespast7years': TotalCreditLinespast7years,
                'TotalTrades': TotalTrades}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
data = pd.read_csv('updated_data.csv')
dn = data[['AvailableBankcardCredit', 'BankcardUtilization', 'BorrowerAPR',
           'BorrowerRate', 'DebtToIncomeRatio', 'DelinquenciesLast7Years',
           'EmploymentStatus', 'EmploymentStatusDuration',
           'EstimatedEffectiveYield', 'EstimatedLoss', 'EstimatedReturn',
           'IncomeRange', 'Investors', 'LenderYield', 'LoanOriginalAmount',
           'LoanOriginationQuarter', 'LoanStatus', 'Occupation',
           'OpenRevolvingMonthlyPayment', 'ProsperRating (numeric)',
           'ProsperScore', 'RevolvingCreditBalance', 'StatedMonthlyIncome', 'Term',
           'TotalCreditLinespast7years', 'TotalTrades']]
df_new = dn.drop(columns=['LoanStatus'])
df = pd.concat([input_df, df_new], axis=0)

# Listing the columns with object datatype
col = df.dtypes[df.dtypes == 'object'].index
# Label Encoding the variables
LE = LabelEncoder()
for cl in col:
    df[cl] = LE.fit_transform(df[cl])

df = df[:1]  # Selects only the first row (the user input data)

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
Loan_Status = np.array(['High Risk', 'Accepted'])
st.write(Loan_Status[prediction])

st.subheader('Prediction Probability')
st.write('0 = High Risk | 1 = Accepted')
st.write(prediction_proba)

if prediction:
    st.subheader('Return on Investment (ROI)')
    NetIncome = input_df['LoanOriginalAmount'] * input_df['BorrowerAPR'] * input_df['Term'] / 12
    ans = (NetIncome * 100 / input_df['LoanOriginalAmount']).astype(str)
    # ans = '{0:.{1}f}'.format(ans)
    st.write('ROI When fees are not taken into account :', ans)
    NetIncome2 = input_df['LoanOriginalAmount'] * input_df['EstimatedEffectiveYield'] * input_df['Term'] / 12
    ans2 = NetIncome2 * 100 / input_df['LoanOriginalAmount']
    st.success('ROI When fees are taken into account :{}'.format(ans2))
