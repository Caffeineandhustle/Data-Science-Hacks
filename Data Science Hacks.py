#!/usr/bin/env python
# coding: utf-8

# In[30]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 1px  black solid !important;\n  color: black !important;\n}\n</style>')


# # Pandas Apply

# In[13]:


import pandas as pd


# In[14]:


ytdata= pd.read_csv('/Users/priyeshkucchu/Desktop/USvideos.csv') 


# In[15]:


def missing_values(x):
    return sum(x.isnull())


# In[16]:


print(" Missing values in each column :")
ytdata.apply(missing_values,axis=0)


# In[17]:


print(" Missing values in each row :")
ytdata.apply(missing_values,axis=1).head()


# # Pandas Count 

# In[18]:


#Count no of data points in each column
ytdata.count(axis=0)


# In[19]:


#No of null data points in Desription column
ytdata.description.isnull().value_counts()


# # Pandas Boolean Indexing

# In[20]:


#Show only those rows where category_id is 24 and no of likes is greater than 12000
ytdata.loc[(ytdata['category_id']==24)& (ytdata['likes']>12000),["category_id","likes"]].head()


# # Pandas Pivot Table

# In[21]:


# import pandas
import pandas as pd
import numpy as np


# In[22]:


#Import dataset
loan = pd.read_csv('/Users/priyeshkucchu/Desktop/loan_train.csv', index_col = 'Loan_ID')


# In[23]:


loan.head()


# In[29]:


pivot = loan.pivot_table(values = ['LoanAmount'],
                         index = ['Gender', 'Married','Dependents', 'Self_Employed'], aggfunc = np.median)
pivot


# # Pandas Crosstab

# In[31]:


# import pandas
import pandas as pd


# In[32]:


#Import dataset
data = pd.read_csv('/Users/priyeshkucchu/Desktop/loan_train.csv', index_col = 'Loan_ID')


# In[35]:


pd.crosstab(data["Credit_History"],data["Self_Employed"],margins=True, normalize = False)


# # Pandas str.split

# In[36]:


import pandas as pd


# In[40]:


# create a dataframe
df = pd.DataFrame({'Person_name':['Naina Chaturvedi', 'Alvaro Morte', 'Alex Pina', 'Steve Jobs']})
df


# In[41]:


# extract first name and last name
df['first_name'] = df['Person_name'].str.split(' ', expand = True)[0]
df['last_name'] = df['Person_name'].str.split(' ', expand = True)[1]

df


# # Extract E-mail from text

# In[43]:


import re
Enquiries_text = 'For any enquiries or feedback related to our product,service, marketing promotions or other general support matters. contactus@samsung.com’'


# In[44]:


re.findall(r"([\w.-]+@[\w.-]+)", Enquiries_text)


# # Removing Emojis from Text

# In[47]:


Emoji_text = 'For example, 🤓🏃‍🏢 could mean “Iam running to work.”'
final_text=Emoji_text.encode('ascii', 'ignore').decode('ascii')

print("Raw tweet with Emoji:",Emoji_text)  
print("Final tweet withput Emoji:",pp_text) 



# # Apply Pandas Operations in Parallel

# In[6]:


get_ipython().system('pip install pandarallel')


# In[7]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
import random
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()


# In[20]:


pandarallel.initialize(progress_bar=True)


# In[33]:


df = pd.DataFrame({
    'A' : [random.randint(8,15) for i in range(1,100000) ],
    'B' : [random.randint(10,20) for i in range(1,100000) ]
})


# In[34]:


def trigono(x):
    return math.sin(x.A**2) + math.sin(x.B**2) + math.tan(x.A**2)


# In[35]:


#without parallelization
%%time
first = df.progress_apply(trigono, axis=1)


# In[36]:


#with parallelization
%%time
first_parallel = df.parallel_apply(trigono, axis=1)


# # Image Augmentation

# In[48]:


# importing all the required libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import skimage.io as io
from skimage.transform import rotate
import numpy as np
import matplotlib.pyplot as plt


# In[53]:


img= io.imread('/Users/priyeshkucchu/Desktop/image.jpeg')


# In[54]:


def augment_img(img):
    fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(22,12))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(rotate(img, angle=45, mode = 'wrap'))
    ax[1].axis('off')
    ax[2].imshow(np.fliplr(img))
    ax[2].axis('off')
    ax[3].imshow(np.flipud(img))
    ax[3].axis('off')
    ax[4].imshow(np.rot90(img))
    ax[4].axis('off')
       
augment_img(img)


# # Pandas melt

# In[55]:


import pandas as pd 


# In[64]:


df = pd.DataFrame({'Person_Name': {0: 'Naina', 1: 'Alex', 2: 'Avarto'}, 
                   'CourseName': {0: 'Masters', 1: 'Graduate', 2: 'Graduate'}, 
                   'Age': {0: 27, 1: 20, 2: 22}}) 


# In[65]:


df


# In[68]:


m1= pd.melt(df, id_vars =['Person_Name'], value_vars =['CourseName', 'Age'])
m1


# In[70]:


m2= pd.melt(df, id_vars =['Person_Name'], value_vars =['Age'])
m2


# # Extract Continuous and categorical data

# In[2]:


import pandas as pd


# In[4]:


#import the dataset
Loan_data = pd.read_csv('/Users/priyeshkucchu/Desktop/loan_train.csv')
Loan_data.shape


# In[5]:


#check data types of column
Loan_data.dtypes


# In[6]:


# Dataframe containing only categorical variable
categorical_variables = Loan_data.select_dtypes("object").head()
categorical_variables.head()


# In[9]:


# Dataframe containing only int variable
integer_variables = Loan_data.select_dtypes("int64")
integer_variables.head()


# In[10]:


# Dataframe containing only number variable
numeric_variables = Loan_data.select_dtypes("number")
numeric_variables.head()


# # Pandas Profiling

# In[8]:


pip install pandas-profiling


# In[4]:


import pandas as pd
import pandas_profiling


# In[2]:


#import dataset
Youtube_data = pd.read_csv('/Users/priyeshkucchu/Desktop/USvideos.csv')


# In[6]:


profiling_report = pandas_profiling.ProfileReport(Youtube_data)


# In[7]:


profiling_report


# In[8]:


# list of all commands
get_ipython().run_line_magic('history', '')


# #  Ipython Interactive Shell
# 

# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


import pandas as pd
data = pd.read_csv('/Users/priyeshkucchu/Desktop/loan_train.csv')


# In[7]:


data.shape
data.head()
data.dtypes
data.info()


# # Parse dates in read_csv() to change data type to datetime

# In[23]:


import pandas as pd


# In[31]:


crime_data = pd.read_csv("/Users/priyeshkucchu/Desktop/crime.csv", engine='python')


# In[32]:


crime_data.dtypes


# In[34]:


crime_data.head()


# In[41]:


#Parse Dates in read_csv()
crime_data = pd.read_csv("/Users/priyeshkucchu/Desktop/crime.csv", engine='python'                         ,parse_dates = ["OCCURRED_ON_DATE"])


# In[43]:


crime_data.dtypes


# # Date Parser

# In[44]:


import datetime
import dateutil.parser


# In[48]:


input_date = '04th Dec 2020'
parsed_date = dateutil.parser.parse(input_date)


# In[52]:


op_date = datetime.datetime.strftime(parsed_date, '%Y-%m-%d')

print(op_date)


# # Inverting a Dictionary

# In[58]:


# Test Dictionary
l_dict = {'Person_Name':'Naina',
           'Age' : 27,
           'Profession' : 'Software Engineer'
           }


# In[59]:


l_dict


# In[60]:


# invert dictionary
invert_dict = {v:k for k,v in l_dict.items()}
invert_dict


# # Pretty Dictionaries 
# 
# 

# In[3]:


l_dict = {'Student_ID': 4,'Student_name' : 'Naina', 'Class_Name': '12th' ,
          'Student_marks' : {'maths' : 92,
                            'science' : 95,
                            'computer science' : 100,
                            'English' : 91}
          }


# In[4]:


l_dict


# In[5]:


# with pprint
import pprint
pprint.pprint(l_dict)


# # Convert List of list to list

# In[6]:


import itertools


# In[7]:


nested_list = [['Naina'], ['Alex', 'Rhody'], ['Sharron', 'Avarto', 'Grace']]
nested_list


# In[8]:


converted_list = list(itertools.chain.from_iterable(nested_list))

print(converted_list)


# In[ ]:




