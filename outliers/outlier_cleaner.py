#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).
        
        predictions is a list of predicted targets that come from your regression, 
        ages is the list of ages in the training set
        net_worths is the actual value of the net worths in the training set. 

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    baseList = []
    for i in range(0, len(predictions)):
        error = np.abs(predictions[i] - net_worths[i])
        baseList.append([ages[i],net_worths[i],error])
           
    baseList.sort(key=lambda tup: tup[2])    
    cleaned_data = np.array(baseList[:81])

    ### your code goes here

    
    return cleaned_data

