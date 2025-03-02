import numpy as np
import pandas as pd 
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    return data

def eda_moty_hair_eye_color_choice_distribution(moty_df, award):
    '''plot count of hair and eye color based on choice of award'''
    sns.catplot(
    data=moty_df,
    x="hair_color",
    hue="eye_color",
    col="choice",
    kind="count",
    height=5,
    aspect=1.5)
   #title

    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of type of awards based on hair and eye color')
   
        plt.savefig('images/moty_hair_eye_color_choice_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of type of awards based on hair and eye color')

        plt.savefig('images/moty_bs_hair_eye_color_choice_distribution.png')

def eda_moty_hair_eye_color_num_achievements_distribution(moty_df, award):
    '''plot median num of achievements based on hair and eye color'''
    
    sns.catplot(
     data=moty_df,
     x="hair_color",
     hue="eye_color",
     y="num_achievements",
     kind="bar",
     height=5,
     aspect=1.5, estimator=np.median)
    
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of achievements based on hair and eye color')

        plt.savefig('images/moty_hair_eye_color_achievements_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of median num of achievements based on hair and eye color')

        plt.savefig('images/moty_bs_hair_eye_color_achievements_distribution.png')


def eda_moty_hair_eye_color_num_runway_shows_distribution(moty_df, award):
    '''plot median num of runway shows based on hair and eye color'''
    
    sns.catplot(
     data=moty_df,
     x="hair_color",
     hue="eye_color",
     y="number_of_runway_shows",
     kind="bar",
     height=5,
     aspect=1.5, estimator=np.median)
    
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of runway shows based on hair and eye color')

        plt.savefig('images/moty_hair_eye_color_runway_shows_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of median num of runway shows based on hair and eye color')

        plt.savefig('images/moty_bs_hair_eye_color_runway_shows_distribution.png')

def eda_num_runway_shows_per_gender(moty_df, award):
    '''plot median num of runway shows per gender'''

    sns.catplot(
     data=moty_df,
     x = 'gender',
        y="number_of_runway_shows",
        kind="bar",
        height=5,
        aspect=1.5, estimator=np.median)
    
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of runway shows')
        plt.savefig('images/moty_num_runway_shows_gender_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of median num of runway shows')
        plt.savefig('images/moty_bs_number_runway_shows_gender_distribution.png')

def eda_gender_distribution_over_years(moty_df):
    '''plot gender distribution over the years'''
    gender_per_yr = moty_df.groupby(['year', 'award', 'gender']).size().reset_index(name='count')

    #mapping the categories to colors
    color_map = {"M": "blue", "F": "red","NB":"green", "TF":"purple", "TM":"orange"}

    categories_pivot = gender_per_yr.pivot_table(index=['year', 'award'], columns = 'gender', values = 'count').fillna(0)
    categories_pivot.plot(kind='bar', stacked=True, color = [color_map.get(x) for x in categories_pivot.columns], figsize=(15, 10))

    plt.title("Gender Distribution over the years")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend(title = "gender")

   
    plt.savefig("images/award_won_by_gender_over_years_matplotlib.png")



def gender_distribution_per_us_agency(agency_df):
    '''gender distribution per agency'''

    gender_cols = ["female","male", "non_binary"]
    for i in gender_cols: 
        if i not in agency_df.columns:
            agency_df[i] = 0

    agency_df[gender_cols] = agency_df[gender_cols].apply(pd.to_numeric, errors='coerce')  

    agency_gender_count = agency_df.groupby("agency_name")[gender_cols].sum().reset_index()
    agency_gender_count.set_index("agency_name", inplace = True)

    bar_width = 0.5  
    fig, ax = plt.subplots(figsize=(18, 8))

    agency_gender_count.plot(kind='bar', ax = ax, width = bar_width)

    plt.title("Gender distribution per agency")
    plt.xlabel("Agency")
    plt.ylabel("Count")
    plt.legend(title = "Gender")


    plt.savefig("images/gender_distribution_per_agency.png")







    
    



   




def main(): 
    
    moty_df = load_data('data/moty.csv')
    print(moty_df.head())
    print("\n")

    agency_df = load_data('data/Agency.csv')
    print(agency_df.head())
    print("\n")

    top_50_female = load_data('data/Top_50_F.csv')
    print(top_50_female.head())
    print("\n")

    top_50_male = load_data('data/Top_50_M.csv')
    print(top_50_male.head())
    print("\n")

    #different categories of awards
    print("Different categories of awards")
    print(moty_df['award'].value_counts())

    #remove white space
    moty_df['name'] = moty_df['name'].str.strip()
    moty_df['award'] = moty_df['award'].str.strip()
    moty_df['hair_color'] = moty_df['hair_color'].str.lower().str.strip()
    moty_df['eye_color'] = moty_df['eye_color'].str.lower().str.strip()
    moty_df['choice'] = moty_df['choice'].str.strip()
    moty_df['gender'] = moty_df['gender'].str.strip()   




    #remove duplicate names 
    moty_df = moty_df.drop_duplicates(subset=['name'], keep='first')

    #preprocess

    #Model of the Year
    moty_df_model_of_the_year = moty_df[moty_df['award'] == 'MOTY']

    #break out star
    moty_bs_df = moty_df[moty_df['award'] == 'BS']
    
    
    print("model of the year breakout star", moty_bs_df.head())
    # print("Distribution of type of awards based on hair and eye color")


     #unique list of models
    # print("Unique list of models")
    # combined_data = pd.concat([moty_df, top_50_male, top_50_female], axis=0)
    # unique_models = combined_data['name'].drop_duplicates().reset_index(drop = True)
    # #write to file
    # unique_models.to_csv('data/unique_models.csv', index = False)
   
    
    
    # print(moty_df['hair_color'].value_counts())
    # print(moty_df['eye_color'].value_counts())
    # print(moty_df['choice'].value_counts())

    # #award = MOTY
    eda_moty_hair_eye_color_choice_distribution(moty_df_model_of_the_year, 'MOTY')
    eda_moty_hair_eye_color_num_achievements_distribution(moty_df_model_of_the_year, 'MOTY')

    # #award = Breakout Star
    eda_moty_hair_eye_color_choice_distribution(moty_bs_df, 'BS')
    eda_moty_hair_eye_color_num_achievements_distribution(moty_bs_df, 'BS')



    #hair eye color number of runway shows
    eda_moty_hair_eye_color_num_runway_shows_distribution(moty_df_model_of_the_year, 'MOTY')
    eda_moty_hair_eye_color_num_runway_shows_distribution(moty_bs_df, 'BS')


    #number  of runway shows per gender
    eda_num_runway_shows_per_gender(moty_df_model_of_the_year, 'MOTY')
    eda_num_runway_shows_per_gender(moty_bs_df, 'BS')

    #plotting over the years gender distribution 
    eda_gender_distribution_over_years(moty_df)


    #gender distribution per US agency
    gender_distribution_per_us_agency(agency_df)


    #TODO: top 50 - female

    #TODO: top 50 - male

    





main()


