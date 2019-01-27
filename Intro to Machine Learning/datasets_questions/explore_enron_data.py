#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print(enron_data['SKILLING JEFFREY K'])

poi_count=0

quantified_email=0
quantified_salary=0
non_quantified_total_payment = 0
total_poi = 0
print(enron_data['FOWLER PEGGY'].keys())
for poi in enron_data.keys():
	if enron_data[poi]["salary"]!='NaN':
		quantified_salary += 1
		
	if enron_data[poi]["email_address"]!='NaN':
		quantified_email += 1
		#print(str(poi)+'  '+str(enron_data[poi]['email_address']))	
	if enron_data[poi]["total_payments"]=='NaN' and enron_data[poi]['poi']==True:
		non_quantified_total_payment += 1
	if enron_data[poi]['poi']==True:
		total_poi += 1
#print('Quantified mail:   '+str(quantified_email))
#print('Quantified salary:   '+str(quantified_salary))
print('Non Quantified total payments:   '+str(non_quantified_total_payment) + ' or ' +str(float(non_quantified_total_payment)/len(enron_data.keys())*100)+'%')
print('Quantified poi:   '+str(total_poi))
'''
print("Skilling   :   "+str(enron_data['SKILLING JEFFREY K']['total_payments']))
print("Kenneth   :   "+str(enron_data['LAY KENNETH L']['total_payments']))
print("Fastow   :   "+str(enron_data['FASTOW ANDREW S']['total_payments']))
'''
