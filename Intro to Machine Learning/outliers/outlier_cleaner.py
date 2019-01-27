#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    import numpy as np
    ### your code goes here 

    error = abs((np.asarray(predictions) - np.asarray(net_worths)))
    
    #print(len(error))
    temp=np.zeros(shape=(90,3))
    
    for i in range(0,len(error),1):
        temp[i][0]=error[i]
        temp[i][1]=ages[i]
        temp[i][2]=net_worths[i]
    #print(len(temp))
    
    list_to_remove = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]

    for k in range(0,9,1):
        maximun = 0
        max_i = 0
        for i in range(0,len(temp),1):
            #print('Temp[i]:  '+ str(temp[i][0]))
            if (temp[i][0] >= maximun and i not in list_to_remove):
                maximun = temp[i][0]
                max_i = i
        list_to_remove.append(max_i)
        list_to_remove.pop(0)       
    #print(list_to_remove)
    #for i in list_to_remove:
        #print(temp[i])
    temp = np.delete(temp,np.asarray(list_to_remove),0)
    cleaned_data=[]
    #print(len(temp))
    for i in range(0,len(temp),1):
        tup=(temp[i][1],temp[i][2],temp[i][0])
        #print(tup)
        cleaned_data.append(tup)
    #print(len(cleaned_data))
    return cleaned_data

