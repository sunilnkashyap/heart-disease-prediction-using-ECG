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
import scipy.signal as signal
from scipy.signal import find_peaks

class ECG:
	def  getImage(self,image):
		"""
		this functions gets user image
		return: user image
		"""
		image=imread(image)
		return image

	def GrayImgae(self,image):
		"""
		This funciton converts the user image to Gray Scale
		return: Gray scale Image
		"""
		image_gray = color.rgb2gray(image)
		image_gray=resize(image_gray,(1572,2213))
		return image_gray

	def DividingLeads(self,image):
		"""
		This Funciton Divides the Ecg image into 13 Leads including long lead. Bipolar limb leads(Leads1,2,3). Augmented unipolar limb leads(aVR,aVF,aVL). Unipolar (+) chest leads(V1,V2,V3,V4,V5,V6)
  		return : List containing all 13 leads divided
		"""
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
		fig.savefig('Leads_1-12_figure.png')
		fig1 , ax1 = plt.subplots()
		fig1.set_size_inches(10, 10)
		ax1.imshow(Lead_13)
		ax1.set_title("Leads 13")
		ax1.axis('off')
		fig1.savefig('Long_Lead_13_figure.png')

		return Leads

	def PreprocessingLeads(self,Leads):
		"""
		This Function Performs preprocessing to on the extracted leads.
		"""
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
		fig2.savefig('Preprossed_Leads_1-12_figure.png')

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
		fig3.savefig('Preprossed_Leads_13_figure.png')


	def SignalExtraction_Scaling(self,Leads):
		"""
		This Function Performs Signal Extraction using various steps,techniques: conver to grayscale, apply gaussian filter, thresholding, perform contouring to extract signal image and then save the image as 1D signal
		"""
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
			if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
			else:
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
	      
		fig4.savefig('Contour_Leads_1-12_figure.png')


	def CombineConvert1Dsignal(self):
		"""
		This function combines all 1D signals of 12 Leads into one FIle csv for model input.
		returns the final dataframe
		"""
		#first read the Lead1 1D signal
		test_final=pd.read_csv('Scaled_1DLead_1.csv')
		location= os.getcwd()
		print(location)
		#loop over all the 11 remaining leads and combine as one dataset using pandas concat
		for files in natsorted(os.listdir(location)):
			if files.endswith(".csv"):
				if files!='Scaled_1DLead_1.csv':
					df=pd.read_csv('{}'.format(files))
					test_final=pd.concat([test_final,df],axis=1,ignore_index=True)

		return test_final
		
	def DimensionalReduciton(self,test_final):
		"""
		This function reduces the dimensinality of the 1D signal using PCA
		returns the final dataframe
		"""
		#first load the trained pca
		pca_loaded_model = joblib.load('pca-final.pkl')
		result = pca_loaded_model.transform(test_final)
		final_df = pd.DataFrame(result)
		return final_df

	def ModelLoad_predict(self, final_df):
		"""
		This Function Loads the pretrained model and perform ECG classification
		return the classification Type.
		"""
		loaded_model = joblib.load('best.pkl')

		# Ensure final_df is 2D
		if final_df.ndim == 1:
			final_df = final_df.reshape(1, -1)

		result = loaded_model.predict(final_df)
		if result[0] == 1:
			return "Your ECG corresponds to Myocardial Infarction"
		elif result[0] == 0:
			return "Your ECG corresponds to Abnormal Heartbeat"
		elif result[0] == 2:
			return "Your ECG is Normal"
		else:
			return "Your ECG corresponds to History of Myocardial Infarction"

	def analyze_pqrs_waves(self, ecg_1dsignal):
		"""
		This function analyzes the P, Q, R, S, T waves of the ECG signal.
		:param ecg_1dsignal: 1D ECG signal
		:return: Dictionary containing P, Q, R, S, T wave features and plot
		"""
		# Convert DataFrame to numpy array
		ecg_signal = ecg_1dsignal.values.flatten()

		# Find R peaks
		r_peaks, _ = find_peaks(ecg_signal, height=0.5, distance=100)

		pqrst_features = {
			'r_peaks': [],
			'p_waves': [],
			'q_waves': [],
			's_waves': [],
			't_waves': []
		}

		for peak in r_peaks:
			# R peak
			pqrst_features['r_peaks'].append((peak, ecg_signal[peak]))

			# P wave (look for maximum in 200ms before R peak)
			p_search_start = max(0, peak - 50)
			p_wave = np.argmax(ecg_signal[p_search_start:peak]) + p_search_start
			pqrst_features['p_waves'].append((p_wave, ecg_signal[p_wave]))

			# Q wave (look for minimum between P wave and R peak)
			q_wave = np.argmin(ecg_signal[p_wave:peak]) + p_wave
			pqrst_features['q_waves'].append((q_wave, ecg_signal[q_wave]))

			# S wave (look for minimum in 100ms after R peak)
			s_search_end = min(len(ecg_signal), peak + 25)
			s_wave = np.argmin(ecg_signal[peak:s_search_end]) + peak
			pqrst_features['s_waves'].append((s_wave, ecg_signal[s_wave]))

			# T wave (look for maximum in 300ms after R peak)
			t_search_end = min(len(ecg_signal), peak + 75)
			t_wave = np.argmax(ecg_signal[s_wave:t_search_end]) + s_wave
			pqrst_features['t_waves'].append((t_wave, ecg_signal[t_wave]))

		# Create PQRST wave diagram
		plt.figure(figsize=(12, 6))
		plt.plot(ecg_signal)
		
		# Plot PQRST points
		for wave_type, color, marker in zip(['p_waves', 'q_waves', 'r_peaks', 's_waves', 't_waves'], 
											['green', 'orange', 'red', 'purple', 'blue'],
											['o', 's', '^', 'D', 'v']):
			x, y = zip(*pqrst_features[wave_type])
			plt.scatter(x, y, c=color, marker=marker, s=100, label=wave_type.capitalize())

		plt.title("ECG Signal with PQRST Waves")
		plt.xlabel("Sample")
		plt.ylabel("Amplitude")
		plt.legend()
		plt.grid(True)
		plt.savefig('pqrst_wave_diagram.png')
		plt.close()

		return pqrst_features

	def DimensionalReduction(self, ecg_1dsignal, pqrst_analysis):
		"""
		This function reduces the dimensionality of the 1D signal using PCA
		and includes PQRST features
		returns the final dataframe and PQRST features
		"""
		# First load the trained pca
		pca_loaded_model = joblib.load('pca-final.pkl')
		result = pca_loaded_model.transform(ecg_1dsignal)
		final_df = pd.DataFrame(result)

		# Ensure final_df is 2D
		if final_df.ndim == 1:
			final_df = final_df.to_frame().T

		# Add PQRST features
		pqrst_features = {
			'mean_r_peak': np.mean(pqrst_analysis['r_peaks']),
			'mean_p_wave': np.mean(pqrst_analysis['p_waves']),
			'mean_q_wave': np.mean(pqrst_analysis['q_waves']),
			'mean_s_wave': np.mean(pqrst_analysis['s_waves']),
			'mean_t_wave': np.mean(pqrst_analysis['t_waves']),
		}

		return final_df, pqrst_features



