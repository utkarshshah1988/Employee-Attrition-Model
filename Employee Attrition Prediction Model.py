#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import gradio as gr
import warnings
warnings.simplefilter("ignore")


# In[2]:


# load model from pickle file

model_pkl_file = 'emp_attrition_model.pkl' 

with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)


# In[3]:


def get_obj_col(df):
    return list(df.select_dtypes(include='object').columns)


# In[4]:


def get_cat_col(df):
    return list(df.select_dtypes(include='category').columns)


# In[5]:


#Convert Object type columns into categorical type
def convert_obj_cat(df):
    obj_col = get_obj_col(df)
    df[obj_col] = df[obj_col].astype('category')
    return df


# In[6]:


def encode_data(df):        
    #le = LabelEncoder()
    cat_col = get_cat_col(df)
    for col in cat_col:
        lable_file = col + 'label_encoder.pkl'
        #print(lable_file)
        
        with open(lable_file, 'rb') as file:  
            le = pickle.load(file)
        #print(le.classes_)
        df[col] = le.transform(df[col])
    return df


# In[7]:


def transform_data_clean(df):
    return encode_data(convert_obj_cat(df))


# In[8]:


def predict(Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField,EnvironmentSatisfaction, Gender, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager):
    # Create a dictionary of the input values
    input_dict = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction":EnvironmentSatisfaction,
        "Gender": Gender,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome":MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime":OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction":RelationshipSatisfaction,
        "StockOptionLevel":StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear":TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }
    
    # Create a pandas dataframe from the input dictionary
    df = pd.DataFrame(input_dict, index=[0])
    df = transform_data_clean(df)
    
    lable_file = 'Attritionlabel_encoder.pkl'
    with open(lable_file, 'rb') as file:  
            le = pickle.load(file)
    result = np.ravel(model.predict(df))
    
    lable_file = 'Attritionlabel_encoder.pkl'
    with open(lable_file, 'rb') as file:
        le = pickle.load(file)
    
    result = le.classes_[result]
    
    
    return result


# In[ ]:





# In[9]:


inputs = [
    gr.components.Number(label="Age"),
    gr.components.Dropdown(choices=["Travel_Rarely", "Travel_Frequently", "Non-Travel"], label="BusinessTravel"),
    gr.components.Number(label="DailyRate"),
    gr.components.Dropdown(choices=["Sales","Research & Development","Human Resources"], label="Department"),
    gr.components.Number(label="DistanceFromHome"),
    gr.components.Dropdown(choices=["Below College","College","Bachelor","Master","Doctor"], label="Education"),
    gr.components.Dropdown(choices=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], label="Education Field"),
    gr.components.Dropdown(choices=[1, 2, 3, 4, 5], label="EnvironmentSatisfaction"),
    gr.components.Dropdown(choices=["Male", "Female"], label="Gender"),
    gr.components.Dropdown(choices=[1, 2, 3, 4], label="JobInvolvement"),
    gr.components.Dropdown(choices=[1, 2, 3, 4, 5], label="Job Level"),
    gr.components.Dropdown(choices=["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"], label="Job Role"),
    gr.components.Dropdown(choices=[1, 2, 3, 4], label="JobSatisfaction"),
    gr.components.Dropdown(choices=["Single", "Married", "Divorced"], label="Marital Status"),
    gr.components.Number(label="MonthlyIncome"),
    gr.components.Number(label="No of Companies Worked"),
    gr.components.Dropdown(choices=["Yes","No"], label="OverTime"),
    gr.components.Number(label="Last Percent Salary Hike"),
    gr.components.Dropdown(choices=[1, 2, 3], label="Last Performance Rating"),
    gr.components.Dropdown(choices=[1, 2, 3, 4], label="RelationshipSatisfaction"),
    gr.components.Dropdown(choices=[0,1, 2, 3], label="StockOptionLevel"),
    gr.components.Number(label="TrainingTimesLastYear"),
    gr.components.Number(label="TotalWorkingYears"),
    gr.components.Dropdown(choices=["Bad","Good","Better","Best"], label="Work-Life Balance"),
    gr.components.Number(label="Years At Company"),
    gr.components.Number(label="Years In CurrentRole"),
    gr.components.Number(label="Years Since Last romotion"),
    gr.components.Number(label="Years With Current Manager")
]

examples =  [
[41, "Travel_Rarely", 1102, "Sales", 1, 2, "Life Sciences", 2, "Female", 3, 2, "Sales Executive", 4, "Single", 5993, 8, "Yes", 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5],
[49, "Travel_Frequently", 279, "Research & Development", 8, 1, "Life Sciences", 3, "Male", 2, 2, "Research Scientist", 2, "Married", 5130, 1, "No", 23, 4, 4, 1, 10, 3, 3, 10, 7, 1, 7],
[37, "Travel_Rarely", 1373, "Research & Development", 2, 2, "Other", 4, "Male", 2, 1, "Laboratory Technician", 3, "Single", 2090, 6, "Yes", 15, 3, 2, 0, 7, 3, 3, 0, 0, 0, 0],
[33, "Travel_Frequently", 1392, "Research & Development", 3, 4, "Life Sciences", 4, "Female", 3, 1, "Research Scientist", 3, "Married", 2909, 1, "Yes", 11, 3, 3, 0, 8, 3, 3, 8, 7, 3, 0],
[27, "Travel_Rarely", 591, "Research & Development", 2, 1, "Medical", 1, "Male", 3, 1, "Laboratory Technician", 2, "Married", 3468, 9, "No", 12, 3, 4, 1, 6, 3, 3, 2, 2, 2, 2],
[32, "Travel_Frequently", 1005, "Research & Development", 2, 2, "Life Sciences", 4, "Male", 3, 1, "Laboratory Technician", 4, "Single", 3068, 0, "No", 13, 3, 3, 0, 8, 2, 2, 7, 7, 3, 6],
[59, "Travel_Rarely", 1324, "Research & Development", 3, 3, "Medical", 3, "Female", 4, 1, "Laboratory Technician", 1, "Married", 2670, 4, "Yes", 20, 4, 1, 3, 12, 3, 2, 1, 0, 0, 0],
[30, "Travel_Rarely", 1358, "Research & Development", 24, 1, "Life Sciences", 4, "Male", 3, 1, "Laboratory Technician", 3, "Divorced", 2693, 1, "No", 22, 4, 2, 1, 1, 2, 3, 1, 0, 0, 0],
[38, "Travel_Frequently", 216, "Research & Development", 23, 3, "Life Sciences", 4, "Male", 2, 3, "Manufacturing Director", 3, "Single", 9526, 0, "No", 21, 4, 2, 0, 10, 2, 3, 9, 7, 1, 8],
[36, "Travel_Rarely", 1299, "Research & Development", 27, 3, "Medical", 3, "Male", 3, 2, "Healthcare Representative", 3, "Married", 5237, 6, "No", 13, 3, 2, 2, 17, 3, 2, 7, 7, 7, 7]
]

    
#    [24,"Travel_Frequently", "Sales",20,"Bachelor","Technical Degree","Female",2,"Manager",2,"Married",8,15,2,15,"Bad",5,3,2,2],
#    [25,"Travel_Rarely", "Research & Development",20,"Bachelor","Marketing","Male",2,"Director",2,"Married",5,12,2,15,"Good",5,3,2,2],
#    [36,"Travel_Rarely", "Sales and Marketing",9,"Bachelor","Marketing","Female",2,"Director",2,"Single",5,12,2,15,"Bad",5,3,2,2],
#    [41,"Travel_Rarely","Plan to Build",1,"College","Engineering","Female",2,"Executive",4,"Single",8,11,1,8,"Bad",6,4,0,5]
#]

outputs = gr.components.Textbox(label="Will Employee Leave Soon?")

title = "Employee Attrition Prediction"
description = "Enter the required inputs to predict employee attrition."

#gr.Examples(examples)
interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=description, examples = examples)
interface.launch()


# In[ ]:




