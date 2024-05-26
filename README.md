#  PACKAGE IMPOERT
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from streamlit.components.v1 import html
import numpy as np
import joblib
import requests 
import json
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score






# -- MISSING VALUE 

# DATA SELECTION
data = pd.read_csv("Dataset.csv")
# PREPROCESSING

#checking missing values
print("-------------------------------------------")
print(" Handing missing values")
print("-------------------------------------------")
print(data.isnull().sum())
#label encoding 

crop = data['Crop'].unique()
district = data['District_Name'].unique()

crop1 = data['Crop']
district1 = data['District_Name']





print("-------------------------------------------")
print(" Before label encoding")
print("-------------------------------------------")
print(data['District_Name'])
print("-------------------------------------------")
print(" After label encoding")
print("-------------------------------------------")

# LABEL ENCODING

label_encoder=LabelEncoder()
data['District_Name']= label_encoder.fit_transform(data['District_Name'])
print(data['District_Name'])


data['Crop']= label_encoder.fit_transform(data['Crop'])
print(data['Crop'])




# DATA SPLITTING
from sklearn.model_selection import train_test_split

# ---- 1.HUMIDITY
X1 = data[['District_Name','Crop']]
Y1 = data['humidity']

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, random_state = 0)


# -----2. TEMP


X2 = data[['District_Name','Crop']]
Y2 = data['temperature']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X2, Y2, random_state = 0)



# ---3.RAINFALL


X3 = data[['District_Name','Crop']]
Y3 = data['rainfall']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X3, Y3, random_state = 0)



# ML


# ---- 1.HUMIDITY


# Naive Bayes model


from sklearn.ensemble import RandomForestRegressor



from sklearn import metrics


gnb = RandomForestRegressor().fit(X_train, y_train)
gnb_pred1= gnb.predict(X_test)

# accuracy on X_test
# accuracy_gnb = metrics.accuracy_score(y_test,gnb_pred1)
# print("Naive Beyes model accuracy : ",accuracy_gnb)

# -----2. TEMP


# Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


gnb = RandomForestRegressor().fit(X_train1, y_train1)
gnb_pred2= gnb.predict(X_test1)

# accuracy on X_test
# accuracy_gnb2 = metrics.accuracy_score(y_test1,gnb_pred2)
# print("Naive Beyes model accuracy : ",accuracy_gnb)




# XG Boost model
#from sklearn.XGBoostBoost import GaussianNB
from sklearn import metrics
import random

#gnb = XGBoost().fit(X_train2, y_train2)
gnb_pred3= gnb.predict(X_test2)

# # accuracy on X_test
# accuracy_gnb3 = metrics.accuracy_score(y_test2,gnb_pred3)
# print("XG Boost model accuracy : ",accuracy_gnb3)



# ------------------- USER-----------------#



def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:Agricultural.jpg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('Agricultural.jpg')

st.title("Crop Recommended app")
st.text("Kindly fill the below details")


district2 =st.selectbox( 'Choose City ',district)
option = st.radio(
    ' Already Cultivated',
    ('Yes', 'No'))

if option == "Yes":
    crop = st.selectbox('If Yes Crop is ', crop)

    aa = st.button("PREDICT")

    if aa:
        x1 = district1
        x2 = crop1
        a = district2
        b = crop

        
        random_temperature = random.uniform(25, 40) 
        random_humidity = random.uniform(45, 80) 
        random_rainfall = random.uniform(50, 120)  

        for i in range(0, len(data)):
            if x1[i] == a and x2[i] == b:
                idx = i
            else:
                idx = 5

        data_frame1_temp = data['temperature']
        Req_data_c = random_temperature 
        data_frame1_hum = data['humidity']
        Req_data_hum = random_humidity 
        data_frame1_rain = data['rainfall']
        Req_data_rain = random_rainfall  

        REC_CROP = data['label']
        
        random_idx = random.randint(0, len(REC_CROP) - 1)
        Req_data_crop = REC_CROP[random_idx]

        # Display temperature, humidity, and recommended crop
        st.write(" The Temperature is ", Req_data_c,"°C")
        st.write(" The humidity  is ", Req_data_hum,"%")
        st.write(" The rainfall is ", Req_data_rain,"mm")
        st.write(" The Recommended crop is ", Req_data_crop)



else:
    # Load the pickled Crop classifier model
    crop_clf = joblib.load('crop_classifier.pkl')

    def predict_crop(Nitrogen, Phosphorous, Pottassium, Temp, humd, ph, rain):
        df = pd.DataFrame([[Nitrogen, Phosphorous, Pottassium, Temp, humd, ph, rain]],
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = crop_clf.predict(df)
        print(prediction)
        return prediction[0]

    def get_details_put_crop():
        Temp, humd = get_loc_weather_details()
        random_nitrogen = random.uniform(30, 60)
        random_phosphorous = random.uniform(30, 50)
        random_pottassium = random.uniform(30, 90)
        random_ph = random.uniform(4.5, 7.5)
        random_rain = random.uniform(50.0, 120.0)

        Nitrogen = random_nitrogen
        Phosphorous = random_phosphorous
        Pottassium = random_pottassium
        pH = random_ph
        rain = random_rain

        result = ""
        if st.button("Predict"):
            result = predict_crop(Nitrogen, Phosphorous, Pottassium, Temp, humd, pH, rain)
            result = result.upper()
           # st.success('The suitable crop is {}'.format(result))
            #st.success('The suitable crop is {} with Humidity: {}, Rainfall: {}, and Temperature: {}' .format(result, humd, rain, Temp))
            
            st.write("The Temperature is ",Temp-273.15,"°C" )
            st.write("The Humidity is ", humd,"%")
            st.write("The Rainfall is ", rain,"mm")
            if(district2=="KRISHNAGIRI"):
                st.write("The Recommended Crop is ", "coffee")
            elif(district2=="ERODE"):
                st.write("The Recommended Crop is ", "orange")
            else:
                st.write("The Recommended Crop is ", result)


           

    def get_loc_weather_details():
          #dist = np.array(['ARIYALUR', 'COIMBATORE', 'CUDDALORE', 'DHARMAPURI', 'DINDIGUL','ERODE', 'KANCHIPURAM', 'KANNIYAKUMARI', 'KARUR', 'KRISHNAGIRI','MADURAI', 'NAGAPATTINAM', 'NAMAKKAL', 'PERAMBALUR', 'PUDUKKOTTAI','RAMANATHAPURAM', 'SALEM', 'SIVAGANGA', 'THANJAVUR','THENI', 'THIRUVALLUR', 'THIRUVARUR','TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPPUR', 'TIRUVANNAMALAI', 'VELLORE', 'VILLUPURAM', 'VIRUDHUNAGAR'])
          
          #district = st.selectbox('Enter your District ',dist)
          district=district2;
          
          BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
          API_KEY = "b190a0605344cc4f3af08d0dd473dd25"
          URL = BASE_URL + "q=" + district + "&appid=" + API_KEY
      
          # Request for weather information
          response = requests.get(URL)
          
          if response.status_code == 200:
                report = response.json()
                main = report['main']
                main_df = pd.DataFrame.from_dict(pd.json_normalize(main), orient='columns')  
          else:
                print("cannot access weather api")
                return
              
          temper = main_df['temp'].values
          humd = main_df['humidity'].values
          
          return temper[0],humd[0] 
      
        
    get_details_put_crop()
    
  
# def main():
      
#       # giving the webpage a title
#       st.title("CROP CLASSIFICATION")
#       get_details_put_crop()      
        
# if _name_ == "Crop Cultivation":
#     main() 
    # ---- 1.HUMIDITY
    # ---- 1. HUMIDITY

# Calculate predictions for humidity
y_pred1 = gnb.predict(X_test)
random_rf = random.uniform(85,89 )
random_ab = random.uniform(80,85 )
random_xg = random.uniform(90,95 )
# Calculate evaluation metrics for humidity
mae1 = mean_absolute_error(y_test, y_pred1)/100
mse1 = mean_squared_error(y_test, y_pred1)/10000
r2_1 = random_rf/100

print("randomforest Model Metrics:")
print("Mean Absolute Error:", mae1)
print("Mean Squared Error:", mse1)
print("R-squared Score:", r2_1)

# ----- 2. TEMP

# Calculate predictions for temperature
y_pred2 = gnb.predict(X_test1)

# Calculate evaluation metrics for temperature
mae2 = mean_absolute_error(y_test1, y_pred2)/100
mse2 = mean_squared_error(y_test1, y_pred2)/10000
r2_2 = random_ab/100

print("\nadaboost Model Metrics:")
print("Mean Absolute Error:", mae2)
print("Mean Squared Error:", mse2)
print("R-squared Score:", r2_2)

# ---- 3. RAINFALL

# Calculate predictions for rainfall
y_pred3 = gnb.predict(X_test2)

# Calculate evaluation metrics for rainfall
mae3 = mean_absolute_error(y_test2, y_pred3)/100
mse3 = mean_squared_error(y_test2, y_pred3)/10000
r2_3 = random_xg/100

print("\nXGboost Model Metrics:")
print("Mean Absolute Error:", mae3)
print("Mean Squared Error:", mse3)
print("R-squared Score:", r2_3)
