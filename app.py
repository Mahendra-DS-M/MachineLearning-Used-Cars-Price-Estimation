# UI Library
import streamlit as st 
from streamlit_option_menu import option_menu

# Base Libraries
import pandas as pd
import joblib

import time

import streamlit as st
from PIL import Image


############# Data ##############
data = pd.read_csv("samplecarsdata.csv")
raw = pd.read_csv("usedcarmodeldata.csv")


########### model ############

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline(steps = [("num_imp",SimpleImputer(strategy = "median")),("Scale",StandardScaler())])
cat_pipe = Pipeline(steps = [("cat_imp",SimpleImputer(strategy = "most_frequent")),("onehot",OneHotEncoder(handle_unknown = 'ignore', sparse = True))])

num_cols = list(raw.select_dtypes(exclude = "object").columns)
del num_cols[-1]
cat_cols = list(raw.select_dtypes(include = "object").columns)

preprocess = ColumnTransformer(transformers = [("num",num_pipe,num_cols),("cat",cat_pipe,cat_cols)], remainder = "passthrough")

# Seperate input(X) and output(y) for modelling

X = raw.drop(['selling_price'],axis = 1)
y = raw['selling_price']

from sklearn.linear_model import LinearRegression

regressor = Pipeline(steps = [("preprocess",preprocess),("LR",LinearRegression())])

regressor.fit(X,y)


########## UI Background ###################

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.wallpapersafari.com/6/34/9zypSX.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()


########################################### UI ##########################
with st.sidebar:
    choose = option_menu("Used Cars Price Estimation", ["Project Info", "Data Studied", "Predictions"],
                         icons=['house', 'table', "tags-fill", 'tags-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Project Info":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About the Project:</p>', unsafe_allow_html=True)    

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">In This Project we had analyzed and got estimations on Used cars. There will be so many factors that can effect the price of a used car like km_driven, mileage, name etc..</p>', unsafe_allow_html=True)
    
    image = Image.open('used-car-valuation.png')

    st.image(image)
        
    image = Image.open('features.PNG')

    st.image(image)

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">In This Project we had analyzed olx used cars data.</p>', unsafe_allow_html=True)

elif choose == "Data Studied":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Data Info:</p>', unsafe_allow_html=True)    

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Data is taken from open-source.</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para">Collected data is having duplicates, special characters & unwanted columns for the analysis. These were handled</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>Sample of Raw Data used for Analysis:</b></p>', unsafe_allow_html=True)
    st.dataframe(data.head())
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>From the above data selling_price is Output column used for analysis and prediction. We can predict the price value by giving other columns data.</b></p>', unsafe_allow_html=True)

elif choose == "Predictions":
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">We can estimate the price value for the mentioned features in data.</p>', unsafe_allow_html=True) 

    ########### Predictions on CSV  Data###########
    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>Predict prices on New data</b></p>', unsafe_allow_html=True)

#### Predictions on Test Data #######    
    testdata = st.file_uploader("Upload data for prediction without selling_price column with the other columns mentioned same as in datainfo:")
    if testdata is not None:
        df = pd.read_csv(testdata)
        st.write("First 5 rows of Uploaded Data:")
        st.write(df)
    else:
        st.write("No Data Given")
    
    if st.button("Predict"):
        with st.spinner("Predicting......"):
            predictions = regressor.predict(df)
            time.sleep(5)
            st.success("Done!.")
            df['selling_price'] = predictions
            st.write("**Generated Predictions.....**")
            st.dataframe(df)
    
    ##### Individual Prediction #####

    st.markdown(""" <style> .para {
        font-size:20px ; font-family: 'Calibri'; color: black; text-align: 'justify'}
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="para"><b>Prediction for a Single Car:</b></p>', unsafe_allow_html=True)

    st.write("Fill out the below form values to give features data to model")

    col1, col2 = st.columns(2)
    with col1:
        cname = st.selectbox("Select Name of Car:", data.name.unique())
    with col2:
        cmodel = st.selectbox("Select Model of Car:", data[data.name == cname].model.unique())
    
    col3, col4 = st.columns(2)
    with col3:
        cyear = st.number_input("Enter Purchase Year:")
    with col4:
        ckm = st.number_input("Enter Number of Kilometers Driven:")
    
    col5, col6 = st.columns(2)
    with col5:
        cfuel = st.selectbox("Select Fuel Type:" , data.fuel.unique())
    with col6:
        ctrans = st.selectbox("Select Transmission Type:" , data.transmission.unique())

    col7, col8 = st.columns(2)
    with col7:
        cowner = st.selectbox("Select Owner Type:" , data.owner.unique())
    with col8:
        cmileage = st.number_input("Enter Mileage Value (Kmpl):")

    col9, col10 = st.columns(2)
    with col9:
        cengine = st.number_input("Enter Engine Capacity Value (Cc) :")
    with col10:
        cmaxp = st.number_input("Enter Max_Power Value (Bhp):")
        
    col11, col12 = st.columns(2)
    with col11:
        ctorque = st.number_input("Enter Torque Value (Nm):")
    with col12:
        cseats = st.number_input("Enter Number of Seats:")


    if st.button("Estimate Price"):
        values = [[cname, cmodel, cyear, ckm, cfuel, ctrans, cowner, cmileage, cengine, cmaxp, ctorque, cseats]]
        check = pd.DataFrame(values, columns = data.columns[0:12])
        st.write("Given Data")
        st.dataframe(check)
        check.fuel.replace({'diesel':4, 'petrol':3, 'lpg':2, 'cng':1}, inplace = True)
        check.transmission.replace({'automatic':2, 'manual':1}, inplace = True)
        check.owner.replace({'firstowner':5, 'secondowner':4, 'thirdowner':3, 'fourthowner':2, 'testdrivecar':1}, inplace = True)
        st.write("**Estimated Selling_Price Value:**")
        price = round(regressor.predict(check)[0])
        price = "₹ " + str(price)
        st.subheader(price)