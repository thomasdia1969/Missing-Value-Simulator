#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[33]:


# df = pd.read_excel("Test_File.xlsx")
df = pd.read_excel("Old_Data.xlsx")
cols = list(df.columns)[0:2]+[0,5,10,15,20,25,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,210,240]
cols
df.columns=cols


# In[34]:


df.head(10)


#make like a video


# This code below essentially makes the row names integers so that they can be used in calculations.

# In[35]:


def fix_middle_1(left,right):
    """
    fix_middle_1 >>> the equation for calculating a missing value when there is
                     only 1 consecutive number missing.
    Input: left >>> the number associated with the smaller temporal value 
           right >>> the number associated with the larger temporal value
    Output: this function returns the simple average between the two numbers.
    """
    return (left+right)/2


# In[36]:


def fix_middle_2(left,right):
    """
    fix_middle_2 >>> the equation for calculating a missing value when there are
                     2 consecutive numbers missing.
    Input: left >>> the number associated with the smaller temporal value 
           right >>> the number associated with the larger temporal value
    Output: this function returns a list of two numbers each 1/3 of the distance
            the next number.
    """
    interval = right-left
    num1 = left+interval/3
    num2 = num1+interval/3
    return [num1,num2]


# In[37]:


def fix_end_1(col_1_y,col_2_y,col_x,col_1_x,col_2_x):
    m = (col_2_y - col_1_y)/(col_2_x - col_1_x)
    b = (col_2_y - (m*col_2_x))
    y3 = (m*col_x)+b
    return y3

#need to delete row if the output is negative

def fix_end_2(col_x_right,col_x_left,col_x_1,col_x_2,col_y_1,col_y_2): #coded for left
    m = (col_y_2 - col_y_1)/(col_x_2 - col_x_1)
    b = (col_y_1 - (m *col_x_1))
    col_y_left = (m*col_x_left)+b
    col_y_right = (m*col_x_right)+b
    return (col_y_left,col_y_right)


# In[39]:


def row_imputer(series,admin_cols=2):
    """
    row_imputer - returns a vector/array of values where the index is the 
                      number of missing 
    
    """
    #pulls out info from series...x/y values
    data = series.iloc[admin_cols:].values
    cols = list(series.index[admin_cols:])

    
    #N is the length of the data
    N = len(data)
    
    #make an array of zeroes of length N
    #counter = np.zeros(N)
    
    #set the counter equal to zero
    cnt = 0 
    
    #returns first the index, then the data value
    for idx,d in enumerate(data):
        
        #if there is an NA value, add 1 to the cnt
        if np.isnan(d):
            cnt+=1
            
            #if one missing and the end is on the right
            if cnt==1 and idx == N-1:                          
                col_y = data[idx] #the y value to predict
                col_1_y = data[idx-1] #one to the left
                col_2_y = data[idx-2] #two to the left
                col_x = cols[idx] #x value
                col_1_x = cols[idx-1] #x value one to the left 
                col_2_x = cols[idx-2] #x value two to the left

                data[idx] = fix_end_1(col_1_y,col_2_y,col_x,col_1_x,col_2_x)

                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])

                
            #if two missing and the end is on the right
            if cnt == 2 and idx == N-1:                         
                col_y_right = data[idx]
                col_y_left = data[idx-1]
                col_y_1 = data[idx-2]
                col_y_2 = data[idx-3]
                col_x_right = cols[idx]
                col_x_left = cols[idx-1]
                col_x_1 = cols[idx-2]
                col_x_2 = cols[idx-3]
                
                vals = fix_end_2(col_x_right,col_x_left,col_x_1,col_x_2,col_y_1,col_y_2)
                i1=idx-1
                i2=idx
                data[i1]=vals[0]
                data[i2]=vals[1]

                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])

        #if there is not an NA value...check to see if 
        #something needs to be imputed (ie cnt>0)
        else:            
            
            ## if one missing from left
            if cnt == 1 and idx == 1:
                col_1_y = data[idx]
                col_2_y = data[idx+1]
                col_x = cols[idx-1]
                col_1_x = cols[idx]
                col_2_x = cols[idx+1]
                
                data[idx-1] = fix_end_1(col_1_y,col_2_y,col_x,col_1_x,col_2_x)
                
                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])

            #if two missing from left end
            if cnt == 2 and idx == 2:
            
                col_x_left = cols[idx-2]
                col_x_right = cols[idx-1]
                
                col_x_1 = cols[idx]
                col_x_2 = cols[idx+1]
                
                col_y_1= data[idx]
                col_y_2 = data[idx+1]
                
                vals = fix_end_2(col_x_right,col_x_left,col_x_1,col_x_2,col_y_1,col_y_2)
                data[idx-2]=vals[0]
                data[idx-1]=vals[1]
                
                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])

                            
            #if missing one data from the middle  
            if cnt == 1 and idx>1 and idx<N: 
                left = data[idx-2]
                right = data[idx]
                
                data[idx-1] = fix_middle_1(left,right)
                
                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])

                
            #if missing 2 consecutive from middle
            if cnt == 2 and idx>2 and idx<N:
                
                left = data[idx-3]
                right = data[idx]
                
                #the value to fill in (calls fix_middle_2 function)
                vals = fix_middle_2(left,right)
                data[idx-1] = vals[1]
                data[idx-2] = vals[0]
                
                series1 = pd.Series(data,index = cols)
                series2 = series.drop(labels = series.index[admin_cols:])
                series = pd.concat([series2,series1])
                
            if cnt >= 200:
                return False

            cnt = 0
    return series


# In[15]:


row_imputer(df.iloc[6])


# Okay, so there are 5 cases we need to test here:
# 
# 1. Two end (right) --> row 1
# 2. Two end (left)  --> row 2 (not working currently)
# 3. One end (left)  --> row 3
# 4. One end (right) --> row 4
# 5. One middle      --> row 5
# 6. Two middle      --> row 6

# In[40]:


def impute_dataframe(df,times,admin_cols=2):
    cols = list(df.columns)[0:admin_cols]+times
    df.columns=cols
    for i in range(len(df)-1,-1,-1):
        series = row_imputer(df.iloc[i])
        if isinstance(series,bool):
            df=df.drop(i)
        else:
            df.iloc[i] = series
    return df


# In[41]:


## times should be the collection times for your data
## admin should be the number of cols that are not computed
times = [0,5,10,15,20,25,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,210,240]
admin_cols = 2
df=impute_dataframe(df,times,admin_cols)

df.head(20)


# In[42]:


df.to_csv("new_thomas.csv")


# In[ ]:





# In[21]:


minimums = []
maximums = []
averages = []

import random
import pandas as pd
import numpy as np
import math
import statistics
for i in range(1000):



    removal_dataset = pd.read_csv("britt_glucose_prepared.csv")
    reference_dataset = pd.read_csv("britt_glucose_prepared.csv")

    num_values_removing = 10

    #rows,columns

    df = removal_dataset
    df3 = reference_dataset
    list_old = []
    cell_indices = []
    list_new = []

    rando_column = []
    rando_row = []
    #column

    count = 0

    for i in range(num_values_removing):

        for i in range(num_values_removing + 50):
            rando = random.randint(2,len(df.columns)-1)
            if len(rando_column) < num_values_removing:
                rando_column.append(rando)
            else:
                pass

        for i in range(num_values_removing + 50):
            rando = random.randint(0,len(df)-1)
            if len(rando_row) < num_values_removing:
                rando_row.append(rando)
            else:
                pass

        random_column = rando_column[count] #come back to this
        random_row = rando_row[count] #come back to this

        #add to the dictionary
        cell_indices.append((random_row, random_column))
        goof = df3.iloc[random_row,random_column]
        list_old.append(goof)
        df.iloc[random_row,random_column]=np.NaN
        count +=1


    new_df=impute_dataframe(df,times,admin_cols)
    new_df=impute_dataframe(new_df,times,admin_cols)


    for i in cell_indices:

        new_values = new_df.iloc[i[0],i[1]]
        list_new.append(new_values)



    differences = []
    count = 0
    for i in list_old:
        diff = abs(list_old[count] - list_new[count])
        differences.append(diff)
        count +=1

    minimum = min(differences)
    maximum = max(differences)
    average = sum(differences)/len(differences)

    minimums.append(minimum)
    maximums.append(maximum)
    averages.append(average)

import math

new_averages = []
for element in averages:
    if not math.isnan(element):
        new_averages.append(element)

new_minimums = []
for element in minimums:
    if not math.isnan(element):
        new_minimums.append(element)

new_maximums = []
for element in maximums:
    if not math.isnan(element):
        new_maximums.append(element)
        
    
mine = sum(new_minimums)/len(new_minimums)
maxe = sum(new_maximums)/len(new_maximums)
ave = sum(new_averages)/len(new_averages)




print("the average minimum value is " + str(mine))
print("the average maximum value is " + str(maxe))
print("the average average value is " + str(ave))
print("the standard deviation of the averages value is " + str(statistics.stdev(new_averages)))



# In[22]:


print("the average minimum value is " + str(mine))
print("the average maximum value is " + str(maxe))
print("the average average value is " + str(ave))
print("the standard deviation of the averages value is " + str(statistics.stdev(new_averages)))


# In[68]:


new_df


# In[ ]:




