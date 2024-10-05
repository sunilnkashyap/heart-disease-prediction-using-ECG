from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class ECG:
	def  getImage(self,image):
		image=imread(image)
		return image

	def GrayImgae(self,image):
		image_gray = color.rgb2gray(image)
		image_gray=resize(image_gray,(1572,2213))
		return image_gray

	def DividingLeads(self,image):
		Lead_1 = image[300:600, 150:643] # Lead 1
		Lead_2 = image[300:600, 646:1135] # Lead aVR
		Lead_3 = image[300:600, 1140:1625] # Lead V1
		Lead_4 = image[300:600, 1630:2125] # Lead V4
		Lead_5 = image[600:900, 150:643] #Lead 2
		Lead_6 = image[600:900, 646:1135] # Lead aVL
		Lead_7 = image[600:900, 1140:1625] # Lead V2
		Lead_8 = image[600:900, 1630:2125] #Lead V5
		Lead_9 = image[900:1200, 150:643] # Lead 3
		Lead_10 = image[900:1200, 646:1135] # Lead aVF
		Lead_11 = image[900:1200, 1140:1625] # Lead V3
		Lead_12 = image[900:1200, 1630:2125] # Lead V6
		Lead_13 = image[1250:1480, 150:2125] # Long Lead

		#All Leads in a list
		Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
		fig , ax = plt.subplots(4,3)
		fig.set_size_inches(10, 10)
		x_counter=0
		y_counter=0

		#Create 12 Lead plot using Matplotlib subplot

		for x,y in enumerate(Leads[:len(Leads)-1]):
			if (x+1)%3==0:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				y_counter+=1
	    
		#save the image
		fig.savefig('processedImages/Leads_1-12_figure.png')
		fig1 , ax1 = plt.subplots()
		fig1.set_size_inches(10, 10)
		ax1.imshow(Lead_13)
		ax1.set_title("Leads 13")
		ax1.axis('off')
		fig1.savefig('processedImages/Long_Lead_13_figure.png')

		return Leads

	def PreprocessingLeads(self,Leads):
		fig2 , ax2 = plt.subplots(4,3)
		fig2.set_size_inches(10, 10)
		#setting counter for plotting based on value
		x_counter=0
		y_counter=0

		for x,y in enumerate(Leads[:len(Leads)-1]):
			#converting to gray scale
			grayscale = color.rgb2gray(y)
			#smoothing image
			blurred_image = gaussian(grayscale, sigma=1)
			#thresholding to distinguish foreground and background
			#using otsu thresholding for getting threshold value
			global_thresh = threshold_otsu(blurred_image)

			#creating binary image based on threshold
			binary_global = blurred_image < global_thresh
			#resize image
			binary_global = resize(binary_global, (300, 450))
			if (x+1)%3==0:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				y_counter+=1
		fig2.savefig('processedImages/Preprossed_Leads_1-12_figure.png')

		#plotting lead 13
		fig3 , ax3 = plt.subplots()
		fig3.set_size_inches(10, 10)
		#converting to gray scale
		grayscale = color.rgb2gray(Leads[-1])
		#smoothing image
		blurred_image = gaussian(grayscale, sigma=1)
		#thresholding to distinguish foreground and background
		#using otsu thresholding for getting threshold value
		global_thresh = threshold_otsu(blurred_image)
		print(global_thresh)
		#creating binary image based on threshold
		binary_global = blurred_image < global_thresh
		ax3.imshow(binary_global,cmap='gray')
		ax3.set_title("Leads 13")
		ax3.axis('off')
		fig3.savefig('processedImages/Preprossed_Leads_13_figure.png')


	def SignalExtraction_Scaling(self,Leads):
		fig4 , ax4 = plt.subplots(4,3)
		#fig4.set_size_inches(10, 10)
		x_counter=0
		y_counter=0
		for x,y in enumerate(Leads[:len(Leads)-1]):
			#converting to gray scale
			grayscale = color.rgb2gray(y)
			#smoothing image
			blurred_image = gaussian(grayscale, sigma=0.7)
			#thresholding to distinguish foreground and background
			#using otsu thresholding for getting threshold value
			global_thresh = threshold_otsu(blurred_image)

			#creating binary image based on threshold
			binary_global = blurred_image < global_thresh
			#resize image
			binary_global = resize(binary_global, (300, 450))
			#finding contours
			contours = measure.find_contours(binary_global,0.8)
			contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
			for contour in contours:
				if contour.shape in contours_shape:
					test = resize(contour, (255, 2))
			if (x+1)%3==0:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				y_counter+=1
	    
			#scaling the data and testing
			lead_no=x
			scaler = MinMaxScaler()
			fit_transform_data = scaler.fit_transform(test)
			Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
			Normalized_Scaled=Normalized_Scaled.T
			#scaled_data to CSV
			if (os.path.isfile('extractedData/scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
				Normalized_Scaled.to_csv('extractedData/Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
			else:
				Normalized_Scaled.to_csv('extractedData/Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
	      
		fig4.savefig('processedImages/Contour_Leads_1-12_figure.png')


	def CombineConvert1Dsignal(self):
		#first read the Lead1 1D signal
		test_final=pd.read_csv('extractedData/Scaled_1DLead_1.csv')
		location= os.getcwd()
		print(location)
		#loop over all the 11 remaining leads and combine as one dataset using pandas concat
		for files in natsorted(os.listdir(location)):
			if files.endswith(".csv"):
				if files!='extractedData/Scaled_1DLead_1.csv':
					df=pd.read_csv('{}'.format(files))
					test_final=pd.concat([test_final,df],axis=1,ignore_index=True)

		return test_final
		
	def DimensionalReduciton(self, test_final):
		# First load the trained pca
		pca_loaded_model = joblib.load('pca-final.pkl')
		
		# Check if the input features match the expected number
		expected_features = 3060  # Set the expected number of features
		current_features = test_final.shape[1]
		
		if current_features != expected_features:
			print(f"Warning: Input has {current_features} features, but PCA expects {expected_features} features.")
			
			if current_features < expected_features:
				# Pad the input with zeros if it has fewer features than expected
				padding = pd.DataFrame(0, index=test_final.index, columns=range(current_features, expected_features))
				test_final = pd.concat([test_final, padding], axis=1)
				print(f"Padded input from {current_features} to {expected_features} features.")
			else:
				# Truncate the input if it has more features than expected
				test_final = test_final.iloc[:, :expected_features]
				print(f"Truncated input from {current_features} to {expected_features} features.")
		
		# Ensure the input has exactly 3060 features
		assert test_final.shape[1] == 3060, f"Input should have 3060 features, but has {test_final.shape[1]}"
		
		# Apply PCA transformation
		result = pca_loaded_model.transform(test_final)
		final_df = pd.DataFrame(result)
		return final_df
	



	
	def ModelLoad_predict(self, final_df):
		loaded_model = joblib.load('best.pkl')
		pca_loaded_model = joblib.load('pca-final.pkl')
		
		# Ensure the input has the correct number of features for PCA
		expected_features = 3060
		current_features = final_df.shape[1]
		
		if current_features != expected_features:
			print(f"Warning: Input has {current_features} features, but PCA expects {expected_features} features.")
			
			if current_features > expected_features:
				final_df = final_df.iloc[:, :expected_features]
				print(f"Truncated input to {expected_features} features.")
			else:
				padding = pd.DataFrame(0, index=final_df.index, columns=range(current_features, expected_features))
				final_df = pd.concat([final_df, padding], axis=1)
				print(f"Padded input from {current_features} to {expected_features} features.")
		
		# Double-check the shape before applying PCA
		if final_df.shape[1] != expected_features:
			raise ValueError(f"Failed to adjust input features. Expected {expected_features}, got {final_df.shape[1]}.")
		
		# Apply PCA transformation
		pca_result = pca_loaded_model.transform(final_df)
		
		result = loaded_model.predict(pca_result)
		if result[0] == 1:
			return "Your ECG corresponds to Myocardial Infarction"
		elif result[0] == 0:
			return "Your ECG corresponds to Abnormal Heartbeat"
		elif result[0] == 2:
			return "Your ECG is Normal"
		else:
			return "Your ECG corresponds to History of Myocardial Infarction"