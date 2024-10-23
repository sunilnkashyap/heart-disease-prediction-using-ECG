import streamlit as st
from Ecg import  ECG
#intialize ecg object
ecg = ECG()
#get the uploaded image
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  """#### **UPLOADED IMAGE**"""
  # call the getimage method
  ecg_user_image_read = ecg.getImage(uploaded_file)
  #show the image
  st.image(ecg_user_image_read)

  """#### **GRAY SCALE IMAGE**"""
  #call the convert Grayscale image method
  ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
  
  #create Streamlit Expander for Gray Scale
  my_expander = st.expander(label='Gray SCALE IMAGE')
  with my_expander: 
    st.image(ecg_user_gray_image_read)
  
  """#### **DIVIDING LEADS**"""
   #call the Divide leads method
  dividing_leads=ecg.DividingLeads(ecg_user_image_read)

  #streamlit expander for dividing leads
  my_expander1 = st.expander(label='DIVIDING LEAD')
  with my_expander1:
    st.image('Leads_1-12_figure.png')
    st.image('Long_Lead_13_figure.png')
  
  """#### **PREPROCESSED LEADS**"""
  #call the preprocessed leads method
  ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)

  #streamlit expander for preprocessed leads
  my_expander2 = st.expander(label='PREPROCESSED LEAD')
  with my_expander2:
    st.image('Preprossed_Leads_1-12_figure.png')
    st.image('Preprossed_Leads_13_figure.png')
  
  """#### **EXTRACTING SIGNALS(1-12)**"""
  #call the sognal extraction method
  ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
  my_expander3 = st.expander(label='CONOTUR LEADS')
  with my_expander3:
    st.image('Contour_Leads_1-12_figure.png')
  
  """#### **CONVERTING TO 1D SIGNAL**"""
  #call the combine and convert to 1D signal method
  ecg_1dsignal = ecg.CombineConvert1Dsignal()
  my_expander4 = st.expander(label='1D Signals')
  with my_expander4:
    st.write(ecg_1dsignal)
    
  """#### **ANALYZE P, Q, R, S, T WAVES**"""
  # Add this new section to analyze P, Q, R, S, T waves
  pqrst_analysis = ecg.analyze_pqrs_waves(ecg_1dsignal)
  my_expander_pqrst = st.expander(label='P, Q, R, S, T Wave Analysis')
  with my_expander_pqrst:
    st.write(pqrst_analysis)
    st.image('pqrst_wave_diagram.png')
  
  """#### **PERFORM DIMENSIONALITY REDUCTION**"""
  # Call the dimensionality reduction function
  ecg_final, pqrst_features = ecg.DimensionalReduction(ecg_1dsignal, pqrst_analysis)
  my_expander5 = st.expander(label='Dimensional Reduction')
  with my_expander5:
    st.write(ecg_final)
    st.write("P, Q, R, S, T Features:", pqrst_features)

  """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
  # Call the Pretrained ML model for prediction
  ecg_model = ecg.ModelLoad_predict(ecg_final)
  my_expander6 = st.expander(label='PREDICTION')
  with my_expander6:
    st.write(ecg_model)
