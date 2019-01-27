#!/usr/bin/python

###
### This function analyses all selected dimensions and prints results
def feature_analysis(features_list,reshape_features, labels):
	import numpy
	from sklearn.linear_model import Lasso

	features_length, features_number = reshape_features.shape
	print("Greetings, mortal. Shall we examine thy feature selection?")
	print("Number of POIs: "+ str(numpy.count_nonzero(labels)))
	print("There are "+ str(features_length) + " entries in "+str(features_number) + " categories, divided as:")
	print("Name \t\t\t\t Total Count \t\t Percentage")
	for dim in range(0,features_number):
		print(features_list[dim+1]+" \t\t\t\t "+str(numpy.count_nonzero(reshape_features[:,dim].astype(float))) + " \t\t "+ str(float((float((numpy.count_nonzero(reshape_features[:,dim].astype(float)))/float(features_length))*100))) + "%")
	lasso = Lasso(alpha=0.1)
	lasso.fit(reshape_features,labels)
	print("Coef:")
	print(lasso.coef_)

###
### This function saves 7 graphs on a pdf file for analysis of outliers and removal.
def first_image_analysis(output_file):
	import matplotlib.backends.backend_pdf
	pdf = matplotlib.backends.backend_pdf.PdfPages(output_file+".pdf")
	plt.ioff()

	# First graph: salary vs total_payments
	fig1 = plt.figure(1)
	plt.title('Salary x Total Payments')
	plt.xlabel('Salary')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig1)
	plt.close(fig1)

	fig2 = plt.figure(2)
	plt.title('Salary x Bonus')
	plt.xlabel('Salary')
	plt.ylabel('Bonus')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,2],marker_size,c=color_labels)
	pdf.savefig(fig2)
	plt.close(fig2)

	fig3 = plt.figure(3)
	plt.title('Salary x Exercised Stock Options')
	plt.xlabel('Salary')
	plt.ylabel('Exercised Stock Options')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,3],marker_size,c=color_labels)
	pdf.savefig(fig3)
	plt.close(fig3)

	fig4 = plt.figure(4)
	plt.title('Salary x Long Term Incentive')
	plt.xlabel('Salary')
	plt.ylabel('Long Term Incentive')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,4],marker_size,c=color_labels)
	pdf.savefig(fig4)
	plt.close(fig4)

	fig5 = plt.figure(5)
	plt.title('Salary x Restricted Stock')
	plt.xlabel('Salary')
	plt.ylabel('Restricted Stock')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,5],marker_size,c=color_labels)
	pdf.savefig(fig5)
	plt.close(fig5)

	fig6 = plt.figure(6)
	plt.title('Salary x Director Fees')
	plt.xlabel('Salary')
	plt.ylabel('Director Fees')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,6],marker_size,c=color_labels)
	pdf.savefig(fig6)
	plt.close(fig6)

	fig7 = plt.figure(7)
	plt.title('From POI to Person x From Person to POI')
	plt.xlabel('From POI to Person')
	plt.ylabel('From Person to POI')
	marker_size=7
	plt.scatter(reshape_features[:,7],reshape_features[:,8],marker_size,c=color_labels)
	pdf.savefig(fig7)
	plt.close(fig7)


	fig8 = plt.figure(8)
	plt.title('From POI to Person x Shared Receipt with POI')
	plt.xlabel('From POI to Person')
	plt.ylabel('Shared Receipt with POI')
	marker_size=7
	plt.scatter(reshape_features[:,7],reshape_features[:,9],marker_size,c=color_labels)
	pdf.savefig(fig8)
	plt.close(fig8)


	fig9 = plt.figure(9)
	plt.title('From Person to POI x Shared Receipt with POI')
	plt.xlabel('From Person to POI')
	plt.ylabel('Shared Receipt with POI')
	marker_size=7
	plt.scatter(reshape_features[:,8],reshape_features[:,9],marker_size,c=color_labels)
	pdf.savefig(fig9)
	plt.close(fig9)

	pdf.close()
	
###
### This function plots 5 graphs to a pdf after outlier removal.
def second_image_analysis(output_file):
	import matplotlib.backends.backend_pdf
	pdf = matplotlib.backends.backend_pdf.PdfPages(output_file+".pdf")
	plt.ioff()

	#Second analysis

	fig8 = plt.figure(8)
	plt.title('Salary x Total Payments')
	plt.xlabel('Salary')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig8)
	plt.close(fig8)

	fig9 = plt.figure(9)
	plt.title('Salary x Bonus')
	plt.xlabel('Salary')
	plt.ylabel('Bonus')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,2],marker_size,c=color_labels)
	pdf.savefig(fig9)
	plt.close(fig9)

	fig10 = plt.figure(10)
	plt.title('Salary x Exercised Stock Options')
	plt.xlabel('Salary')
	plt.ylabel('Exercised Stock Options')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,3],marker_size,c=color_labels)
	pdf.savefig(fig10)
	plt.close(fig10)

	fig11 = plt.figure(11)
	plt.title('Salary x Long Term Incentive')
	plt.xlabel('Salary')
	plt.ylabel('Long Term Incentive')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,4],marker_size,c=color_labels)
	pdf.savefig(fig11)
	plt.close(fig11)

	fig12 = plt.figure(12)
	plt.title('Salary x Restricted Stock')
	plt.xlabel('Salary')
	plt.ylabel('Restricted Stock')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,5],marker_size,c=color_labels)
	pdf.savefig(fig12)
	plt.close(fig12)

	fig13 = plt.figure(13)
	plt.title('Salary x Director Fees')
	plt.xlabel('Salary')
	plt.ylabel('Director Fees')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,6],marker_size,c=color_labels)
	pdf.savefig(fig13)
	plt.close(fig13)

	fig14 = plt.figure(14)
	plt.title('From POI to Person x Shared Receipt with POI')
	plt.xlabel('From POI to Person')
	plt.ylabel('Shared Receipt with POI')
	marker_size=7
	plt.scatter(reshape_features[:,7],reshape_features[:,8],marker_size,c=color_labels)
	pdf.savefig(fig14)
	plt.close(fig14)

	fig18 = plt.figure(18)
	plt.title('Salary x From POI to Person')
	plt.xlabel('Salary')
	plt.ylabel('From POI to Person')
	marker_size=7
	plt.scatter(reshape_features[:,0],reshape_features[:,7],marker_size,c=color_labels)
	pdf.savefig(fig18)
	plt.close(fig18)

	fig19 = plt.figure(19)
	plt.title('Director Fees x From POI to Person')
	plt.xlabel('Director Fees')
	plt.ylabel('From POI to Person')
	marker_size=7
	plt.scatter(reshape_features[:,6],reshape_features[:,7],marker_size,c=color_labels)
	pdf.savefig(fig19)
	plt.close(fig19)

	fig20 = plt.figure(20)
	plt.title('Director Fees x Shared Receipt with POI')
	plt.xlabel('Director Fees')
	plt.ylabel('Shared Receipt with POI')
	marker_size=7
	plt.scatter(reshape_features[:,6],reshape_features[:,8],marker_size,c=color_labels)
	pdf.savefig(fig20)
	plt.close(fig20)

	fig21 = plt.figure(21)
	plt.title('Director Fees x Exercised Stock Options')
	plt.xlabel('Director Fees')
	plt.ylabel('Exercised Stock Options')
	marker_size=7
	plt.scatter(reshape_features[:,6],reshape_features[:,3],marker_size,c=color_labels)
	pdf.savefig(fig21)
	plt.close(fig21)

	fig22 = plt.figure(22)
	plt.title('Bonus x Shared Receipt with POI')
	plt.xlabel('Bonus')
	plt.ylabel('Shared Receipt with POI')
	marker_size=7
	plt.scatter(reshape_features[:,2],reshape_features[:,8],marker_size,c=color_labels)
	pdf.savefig(fig22)
	plt.close(fig22)

	fig23 = plt.figure(23)
	plt.title('Exercised Stock Options x Total Payments')
	plt.xlabel('Exercised Stock Options')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,3],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig23)
	plt.close(fig23)

	fig24 = plt.figure(24)
	plt.title('Restricted Stock x Total Payments')
	plt.xlabel('Restricted Stock')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,5],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig24)
	plt.close(fig24)

	fig25 = plt.figure(25)
	plt.title('From POI to Person x From Person to POI')
	plt.xlabel('From POI to Person')
	plt.ylabel('From Person to POI')
	marker_size=7
	plt.scatter(reshape_features[:,7],reshape_features[:,9],marker_size,c=color_labels)
	pdf.savefig(fig25)
	plt.close(fig25)

	fig26 = plt.figure(26)
	plt.title('Expenses x Total Payments')
	plt.xlabel('Expenses')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,12],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig26)
	plt.close(fig26)

	fig27 = plt.figure(27)
	plt.title('Other x Total Payments')
	plt.xlabel('Other')
	plt.ylabel('Total Payments')
	marker_size=7
	plt.scatter(reshape_features[:,13],reshape_features[:,1],marker_size,c=color_labels)
	pdf.savefig(fig27)
	plt.close(fig27)

	pdf.close()

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','bonus','exercised_stock_options',
'long_term_incentive','restricted_stock','director_fees','from_poi_to_this_person',
'from_this_person_to_poi','shared_receipt_with_poi','to_messages','from_messages',
'expenses','other'] 
# You will need to use more features

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "r") as data_file:	#Windows
with open("final_project_dataset_unix.pkl", "rb") as data_file:		#linux
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict
#print(len(my_dataset))
#print(my_dataset)
import numpy

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
reshape_features=numpy.reshape(features,(len(features),len(features[0])))

### analyses our selected features
#feature_analysis(features_list,reshape_features, labels)

### Apparently we have 145 entries, from which 18 are positive POIs, divided as:
### Name 				 		Count non zeroes  	Percentage of non zeroes
### salary 				 		95 		 			65.5172413793%
### total_payments 		 		125 	 			86.2068965517%
### bonus 				 		82 		 			56.5517241379%
### exercised_stock_options 	102 	 			70.3448275862%
### long_term_incentive 		66 		 			45.5172413793%
### restricted_stock 			110 	 			75.8620689655%
### director_fees 				17 		 			11.724137931%
### from_poi_to_this_person 	74 		 			51.0344827586%
### from_this_person_to_poi 	66 		 			45.5172413793%
### shared_receipt_with_poi 	86 		 			59.3103448276%
### to_messages 				86 		 			59.3103448276%
### from_messages 				86 		 			59.3103448276%
### expenses 				 	95 		 			65.5172413793%
### other 				 		93 		 			64.1379310345%

### Running the Lasso regularizer, we get:
### [ -1.45127819e-07   6.47593124e-09  -2.38642361e-08   2.25498067e-08
###   2.03737161e-08   5.06388695e-09  -1.28738886e-06  -1.00028528e-04
###   5.87189229e-04   2.13469305e-04  -8.11863318e-05  -8.17908593e-06
###   1.05943512e-08  -6.47615246e-08]

### Those are not good results, and the least bad looking features are
### 'director_fees','from_poi_to_this_person','from_this_person_to_poi',
### 'shared_receipt_with_poi'

### Task 2: Remove outliers

import matplotlib.pyplot as plt


### To remove the outliers, we have to plot each dimension first.

## We will crete a support vector to plot color, in which POIs are red and 
### non POIs are blue.
color_labels = []
for label in labels:
	if label:
		color_labels.append('r')
	else:
		color_labels.append('b')

#runs the first image analysis
first_image_analysis('first_image_analysis')


### After visually analyzing our data, it seems only one point needs to be manually
### removed from the Salary comparisons. As for the email metrics, after our initial
### first plot, it seems there is little difference between the from_poi_to_person
### and from_person_to_poi. The shared_receipt_with_poi doesn't need to be used with
### both from_poi and to_poi, so we by visually analyzing the graphs it is observable
### it is preferable to use from_poi because it has a better distribution. There is 
### also one point that could be considered an outlier on the from poi and will be
### removed

### removes outliers from my_dataset
max_salary_index = numpy.argmax(reshape_features[:,0])
max_salary_value = reshape_features[max_salary_index,0]

max_from_poi_index = numpy.argmax(reshape_features[:,7])
max_from_poi_value = reshape_features[max_from_poi_index,7]

for key in my_dataset.keys():
	if(my_dataset[key]['salary']==max_salary_value):
		print('Deleting '+key)
		del my_dataset[key]
	elif(my_dataset[key]['from_poi_to_this_person']==max_from_poi_value):
		print('Deleting '+key)
		del my_dataset[key]

#reloads the features object for further use:
features_list = ['poi','salary','total_payments','bonus','exercised_stock_options',
'long_term_incentive','restricted_stock','director_fees','from_poi_to_this_person',
'shared_receipt_with_poi','from_this_person_to_poi','to_messages','from_messages',
'expenses','other']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data) 

reshape_features=numpy.reshape(features,(len(features),len(features[0])))
color_labels = []
for label in labels:
	if label:
		color_labels.append('r')
	else:
		color_labels.append('b')



#runs the second image analysis
second_image_analysis('second_image_analysis')

### Task 3: Create new feature(s)
from mpl_toolkits import mplot3d

### First we will follow the tip from our instructor and implement the
### percentage of emails from and to POIs.
from_poi_percent = numpy.divide(reshape_features[:,7],reshape_features[:,10]).astype(float)
to_poi_percent = numpy.divide(reshape_features[:,9],reshape_features[:,11]).astype(float)

plt.figure(30)
plt.title('From POI Percent x To POI Percent')
plt.xlabel('From POI Percent')
plt.ylabel('To POI Percent')
marker_size=7
plt.scatter(from_poi_percent,to_poi_percent,marker_size,c=color_labels)
plt.savefig('from_to_poi_percent.png')

### Another idea is to use the dimension shared_receipt.

shared_receipt_percent = numpy.divide(reshape_features[:,8],reshape_features[:,11]).astype(float)

plt.figure(31)
plt.title('From POI Percent x Shared Receipt Percent')
plt.xlabel('From POI Percent')
plt.ylabel('Shared Receipt Percent')
marker_size=7
plt.scatter(from_poi_percent,shared_receipt_percent,marker_size,c=color_labels)
plt.savefig('from_poi_shared_receipt_percent.png')

plt.figure(32)
plt.title('To POI Percent x Shared Receipt Percent')
plt.xlabel('To POI Percent')
plt.ylabel('Shared Receipt Percent')
marker_size=7
plt.scatter(to_poi_percent,shared_receipt_percent,marker_size,c=color_labels)
plt.savefig('to_poi_shared_receipt_percent.png')

### Unfortunately, we see little improvement in this attempt.

### A noticeable thing is that despite the fact that total_payments
### has most of non zeroes, it has a very large value for a POI which
### we do not want to remove. A workaround could be to apply a log function
### to this dimension.
log_total_payments = numpy.add(reshape_features[:,1],1)
log_total_payments = numpy.log(log_total_payments)

plt.figure(25)
plt.title('Salary x Log Total Payments')
plt.xlabel('Salary')
plt.ylabel('Log Total Payments')
marker_size=7
plt.scatter(reshape_features[:,0],log_total_payments,marker_size,c=color_labels)
plt.savefig('log_total_payments1.png')


### It seems one logical option to push all those red spots away from the 
### the blue ones would be to crete a Salary X Shared Receipt and 
### correlate the social interactions. We could try for instance multiplying 
### the shared receipts by the Bonus, for an example.


salaryXreceipt = numpy.multiply(reshape_features[:,0],reshape_features[:,8])
#bonusXreceipt = numpy.multiply(bonusXreceipt,reshape_features[:,7])
salaryXreceipt = numpy.sqrt(numpy.sqrt(salaryXreceipt))
#print(bonusXreceipt)

plt.figure(15)
plt.title('Sqrt2 SalaryXSharedReceipt x Log Total Payments')
plt.xlabel('Log Total Payments')
plt.ylabel('Sqrt2 SalaryXSharedReceipt')
marker_size=7
plt.scatter(log_total_payments,salaryXreceipt,marker_size,c=color_labels)
plt.savefig('salaryXreceipt1.png')

plt.figure(16)
plt.title('Sqrt2 SalaryXSharedReceipt x From POI')
plt.xlabel('From POI')
plt.ylabel('Sqrt2 SalaryXSharedReceipt')
marker_size=7
plt.scatter(reshape_features[:,7],salaryXreceipt,marker_size,c=color_labels)
plt.savefig('salaryXreceipt2.png')

plt.figure(17)
ax = plt.axes(projection='3d')
plt.title('Sqrt2 SalaryXSharedReceipt x Total Payments x From POI')
ax.set_xlabel('From POI')
ax.set_ylabel('Total Payments')
ax.set_zlabel('Sqrt2 SalaryXSharedReceipt')

ax.scatter3D(log_total_payments,reshape_features[:,7],salaryXreceipt,
	s=2,c=color_labels)
plt.savefig('salaryXreceipt3d.png')

### Other similar dimension we could try is restricted_stock and from_poi

restrictedXfrom_poi = numpy.multiply(reshape_features[:,5],reshape_features[:,7])
#bonusXreceipt = numpy.multiply(bonusXreceipt,reshape_features[:,7])
restrictedXfrom_poi = numpy.sqrt(numpy.sqrt(restrictedXfrom_poi))
#print(bonusXreceipt)

plt.figure(26)
plt.title('Sqrt2 RestrictedXFromPOI x Log Total Payments')
plt.xlabel('Log Total Payments')
plt.ylabel('Sqrt2 RestrictedXFromPOI')
marker_size=7
plt.scatter(log_total_payments,restrictedXfrom_poi,marker_size,c=color_labels)
plt.savefig('restrictedXfrom_poi1.png')

plt.figure(27)
plt.title('Sqrt2 RestrictedXFromPOI x Shared Receipt')
plt.xlabel('Shared Receipt')
plt.ylabel('Sqrt2 RestrictedXFromPOI')
marker_size=7
plt.scatter(reshape_features[:,8],restrictedXfrom_poi,marker_size,c=color_labels)
plt.savefig('restrictedXfrom_poi2.png')

plt.figure(28)
ax = plt.axes(projection='3d')
plt.title('Sqrt2 RestrictedXFromPOI x Total Payments x SharedReceipt')
ax.set_xlabel('SharedReceipt')
ax.set_ylabel('Total Payments')
ax.set_zlabel('Sqrt2 RestrictedXFromPOI')

ax.scatter3D(log_total_payments,reshape_features[:,8],restrictedXfrom_poi,
	s=2,c=color_labels)
plt.savefig('restrictedXfrom_poi3d.png')

### Now, we can re-run the Lasso Tool and check how those new dimensions
### correlate to our labels.

new_feature_list = ['poi','from_poi_percent','to_poi_percent','shared_receipt_percent',
 'log_total_payments','salaryXreceipt','restrictedXfrom_poi']

new_features_array = numpy.append(numpy.reshape(from_poi_percent,(len(from_poi_percent),1)),
	numpy.reshape(to_poi_percent,(len(to_poi_percent),1)), axis=1)
new_features_array = numpy.append(new_features_array,
	numpy.reshape(shared_receipt_percent,(len(shared_receipt_percent),1)), axis=1)
new_features_array = numpy.append(new_features_array,
	numpy.reshape(log_total_payments,(len(log_total_payments),1)), axis=1)
new_features_array = numpy.append(new_features_array,
	numpy.reshape(salaryXreceipt,(len(salaryXreceipt),1)), axis=1)
new_features_array = numpy.append(new_features_array,
	numpy.reshape(restrictedXfrom_poi,(len(restrictedXfrom_poi),1)), axis=1)
new_features_array = numpy.nan_to_num(new_features_array)

#runs feature analysis again for the new features:
#feature_analysis(new_feature_list,new_features_array, labels)

### Oddly, the suggested features did not get a nice Lasso coef results.

# Number of POIs: 18
# There are 143 entries in 6 categories, divided as:
# Name 				 			Total Count 		 Percentage

# from_poi_percent 				73 		 			 51.048951049%
# to_poi_percent 				65 		 			 45.4545454545%
# shared_receipt_percent 		85 		 			 59.4405594406%
# log_total_payments 			123 		 		 86.013986014%
# salaryXreceipt 				66 		 			 46.1538461538%
# restrictedXfrom_poi 			66 		 			 46.1538461538%

# Coef:
# [-0.          0.          0.0002659   0.00202666  0.00052639  0.00152021]

#print(reshape_features.shape)
import math

### Now this new dimension must be added to the dataset
for key in my_dataset.keys():
	from_poi_percent_unit=float(my_dataset[key]['from_poi_to_this_person'])/float(my_dataset[key]['to_messages'])
	to_poi_percent_unit=float(my_dataset[key]['from_this_person_to_poi'])/float(my_dataset[key]['from_messages'])
	log_total_payments_unit = math.log(float(float(my_dataset[key]['total_payments'])+1))
	restrictedXfrom_poi_unit = float(math.sqrt(math.sqrt(float(my_dataset[key]['restricted_stock'])*float(my_dataset[key]['from_poi_to_this_person']))))
	salaryXreceipt_unit = float(math.sqrt(math.sqrt(float(my_dataset[key]['salary'])*float(my_dataset[key]['shared_receipt_with_poi']))))
	
	#print(str(log_total_payments_unit)+"  "+str(restrictedXfrom_poi_unit)+"  "+str(bonusXreceipt_unit))
	if math.isnan(salaryXreceipt_unit):
		my_dataset[key]['salaryXreceipt'] = 0
	else:
		my_dataset[key]['salaryXreceipt'] = salaryXreceipt_unit

	if math.isnan(restrictedXfrom_poi_unit):
		my_dataset[key]['restrictedXfrom_poi'] = 0
	else:
		my_dataset[key]['restrictedXfrom_poi'] = float(math.sqrt(math.sqrt(restrictedXfrom_poi_unit)))

	if math.isnan(log_total_payments_unit):
		my_dataset[key]['log_total_payments'] = 0
	else:
		my_dataset[key]['log_total_payments'] = float(log_total_payments_unit)

	if math.isnan(from_poi_percent_unit):
		my_dataset[key]['from_poi_percent'] = 0
	else:
		my_dataset[key]['from_poi_percent'] = float(from_poi_percent_unit)

	if math.isnan(to_poi_percent_unit):
		my_dataset[key]['to_poi_percent'] = 0
	else:
		my_dataset[key]['to_poi_percent'] = float(to_poi_percent_unit)

### Some features with low coef will be removed.

#reloads the features object for further use:
features_list = ['poi','director_fees','from_poi_percent','to_poi_percent',
'restrictedXfrom_poi','log_total_payments','salaryXreceipt']
#features_list = ['poi','salary','from_poi_to_this_person','bonusXreceipt']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data) 

reshape_features=numpy.reshape(features,(len(features),len(features[0])))
#print(reshape_features)
color_labels = []
for label in labels:
	if label:
		color_labels.append('r')
	else:
		color_labels.append('b')

### Now we will apply PCA and reduce some old features and create new ones.
from sklearn.preprocessing import StandardScaler
#first we will scale the dimensions
scaler = StandardScaler()
scaled_features = scaler.fit_transform(reshape_features[:])

from sklearn.decomposition import PCA


pca2d = PCA(n_components=2)
principalComponents2d = pca2d.fit_transform(scaled_features)

###prints PCA 2D
plt.figure(13)
plt.title('Principal Component 1 x Principal Component 2')
plt.xlabel('PC1')
plt.ylabel('PC2')
marker_size=7
plt.scatter(principalComponents2d[:,0],principalComponents2d[:,1],marker_size,c=color_labels)
plt.savefig('pca2d.png')

### After analyzing the PCA for two dimensions, it did not produce a nice separated
### group of features, se I decided to risk a 3D principal component to see what I
### could get.

pca3d = PCA(n_components=3)
principalComponents3d = pca3d.fit_transform(scaled_features)

###prints PCA 3D
plt.figure(14)
ax = plt.axes(projection='3d')
plt.title('Principal Component 1 x Principal Component 2 x Principal Component 3')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.scatter3D(principalComponents3d[:,1],principalComponents3d[:,0],principalComponents3d[:,2],
	s=2,c=color_labels)
plt.savefig('pca3d.png')

### It might be worth a shot since it seems a nice nucleus of POI was formed.

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


scaler_gaussian = StandardScaler()
pca_gaussian = PCA(n_components=3)
gaussian = GaussianNB()

pipe_gaussian = Pipeline([('scaler',scaler_gaussian),('pca',pca_gaussian),('gaussian', gaussian)])
pipe_gaussian.fit(reshape_features[:],labels)

scaler_svc = StandardScaler()
pca_svc = PCA(n_components=3)
svc_step = SVC(kernel='rbf')
pipe_svc = Pipeline([('scaler',scaler_svc),('pca',pca_svc),('svc', svc_step)])

param_grid = {
    'svc__C':[0.0000001,0.00001,0.001,0.05,0.1,0.5,1.0,10,100,1000,10000],
    'svc__gamma':[0.000000001,0.000001,0.0005,0.001,0.05,0.1,1,10,100,1000]
}
#pipe_svc.fit(reshape_features[:],labels)
grid = GridSearchCV(pipe_svc, cv=3, param_grid=param_grid)
grid.fit(reshape_features[:],labels)


print("Best: %f using %s" % (grid.best_score_, 
    grid.best_params_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
'''
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#print('Feature Length:   '+ str(len(features[0])))
#print(features_test)
y_preds = pipe_gaussian.predict(features_test)

from sklearn.metrics import precision_recall_fscore_support
#print(labels_test)
#print(y_preds)

print('GaussianNB: '+str(precision_recall_fscore_support(labels_test, y_preds,average='binary')))

y_preds = grid.predict(features_test)
#print(y_preds)
print('SVC: '+str(precision_recall_fscore_support(labels_test, y_preds,average='binary')))

import tester

print("Testing GaussianNB:")
tester.test_classifier(pipe_gaussian,my_dataset,features_list)

scaler_svc = StandardScaler()
pca_svc = PCA(n_components=2)
svc = SVC(C=1000,kernel='rbf',gamma=1)
pipe_svc = Pipeline([('scaler',scaler_svc),('pca',pca_svc),('svc', svc)])
pipe_svc.fit(reshape_features[:],labels)


print("Testing SVC:")
tester.test_classifier(pipe_svc,my_dataset,features_list)



### After comparing the results, the estimator of choice will be the SVM
### with RBF kernel and C=1000, gamma =1. We've got Precision: 0.43675 and
### Recall: 0.36250 against P:0.30776 and R:0.351 from the GaussianNB.

clf = pipe_svc

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

