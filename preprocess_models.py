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


def import_skintone_data():
    skintones_first75 = pd.read_csv('data/skintones_first75.csv')
    skintones_last75 = pd.read_csv('data/skintones_last75.csv')
    skintones_first75.rename(columns={'model name': 'name'}, inplace=True)
    skintones = pd.concat([skintones_first75, skintones_last75], axis=0)
    skintones = skintones.drop_duplicates(subset=['name'], keep='first')
    if 'Unnamed: 0' in skintones.columns:
        skintones = skintones.drop(columns=['Unnamed: 0'])
    # split RGB values into separate columns
    skintones[['R', 'G', 'B']] = skintones['RGB_value'].str.split(',', expand=True).astype(int)
    # drop the original RGB_value column
    skintones = skintones.drop(columns=['RGB_value'])
    skintones = skintones.reset_index(drop=True)
    skintones.to_csv('data/skintones.csv', index = False)
    return skintones


def plot_skintone_distribution():
    skintones_df = load_data('data/skintones.csv')
    skintones_df['R'] = skintones_df['R'].astype(int)
    skintones_df['G'] = skintones_df['G'].astype(int)
    skintones_df['B'] = skintones_df['B'].astype(int)

    # Create a bar plot of the skintone distribution
    plt.figure(figsize=(10, 8))
    skintone_counts = skintones_df['skin_tone'].value_counts().sort_index()
    print(skintone_counts)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    plt.bar(x=list(skintone_counts.index), height=list(skintone_counts.values), color=[skin_tone_colormap[i] for i in skintone_counts.index])
    plt.title('Skintone Distribution')
    plt.xlabel('Skin Tone on Monk Skin Tone Scale')
    plt.ylabel('Count')
    plt.xticks(list(range(1, 11)), rotation=90)
    plt.savefig('images/skintone_distribution_barchart.png')
    plt.show()

def plot_skintone_vs_achievements():
    skintones_df = load_data('data/skintones.csv')
    moty_df = load_data('data/moty.csv')
    combined_data = pd.merge(skintones_df, moty_df, on='name', how='inner')
    print(combined_data.head())
    combined_data.rename(columns={'skin_tone_x': 'skin_tone', 'skin_tone_y': 'skin_tone_moty'}, inplace=True)
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty'])

    # plot mean number of achievements per skintone
    plt.figure(figsize=(10, 8))
    combined_data['skin_tone'] = combined_data['skin_tone'].astype(int)
    mean_achievements = combined_data.groupby('skin_tone')['num_achievements'].mean()
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    plt.bar(x=list(mean_achievements.index), height=list(mean_achievements.values), color=[skin_tone_colormap[i] for i in mean_achievements.index])
    plt.title('Mean Number of Achievements per Skintone')
    plt.xlabel('Skin Tone on Monk Skin Tone Scale')
    plt.ylabel('Mean Number of Achievements')
    plt.xticks(list(range(1, 11)), rotation=90)
    plt.savefig('images/skintone_vs_achievements.png')
    plt.show()

    
def moty_winners_by_skintone(moty_df_model_of_the_year, skintones_df):
    plt.figure(figsize=(10, 8))
    combined_data = pd.merge(moty_df_model_of_the_year, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty'})
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty'])
    combined_data['skin_tone'] = combined_data['skin_tone'].astype(int)
    skintone_counts = combined_data['skin_tone'].value_counts().sort_index()
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    # keep only award == 'MOTY'
    plt.bar(x=list(skintone_counts.index), height=list(skintone_counts.values), color=[skin_tone_colormap[i] for i in skintone_counts.index])
    plt.title('MOTY Winners by Skintone')
    plt.xlabel('Skin Tone on Monk Skin Tone Scale')
    plt.ylabel('Count')
    plt.xticks(list(range(1, 11)), rotation=90)
    plt.savefig('images/moty_winners_by_skintone.png')
    plt.show()


def skintone_vs_gender(skintones_df, moty_df, top_50_male, top_50_female):
    models = pd.concat([moty_df, top_50_female, top_50_male], axis=0)
    models = models.loc[:, ['name', 'gender']].drop_duplicates().reset_index(drop = True)
    print(models.head())
    combined_data = pd.merge(skintones_df, models, on='name', how='inner')
    combined_data = combined_data.drop(columns=['R', 'G', 'B'])
    combined_data['skin_tone'] = combined_data['skin_tone'].astype(int)
    grouped_data = combined_data.groupby('gender')['skin_tone'].value_counts().unstack().fillna(0)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    fig, ax = plt.subplots(1, 5, figsize=(100, 5), sharey=True, )
    # each plot is a gender category, with the number of models in each skintone category

    ax[0].bar(x=grouped_data.columns, height=grouped_data.iloc[0], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[0].set_title('Cisgender Female Models by Skintone')
    ax[0].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(list(range(1, 11)))
    ax[0].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[1].bar(x=grouped_data.columns, height=grouped_data.iloc[1], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[1].set_title('Cisgender Male Models by Skintone')
    ax[1].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[1].set_ylabel('Count')
    ax[1].set_xticks(list(range(1, 11)))
    ax[1].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[2].bar(x=grouped_data.columns, height=grouped_data.iloc[2], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[2].set_title('Non-Binary Models by Skintone')
    ax[2].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[2].set_ylabel('Count')
    ax[2].set_xticks(list(range(1, 11)))
    ax[2].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[3].bar(x=grouped_data.columns, height=grouped_data.iloc[3], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[3].set_title('Transgender Female Models by Skintone')
    ax[3].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[3].set_ylabel('Count')
    ax[3].set_xticks(list(range(1, 11)))
    ax[3].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[4].bar(x=grouped_data.columns, height=grouped_data.iloc[4], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[4].set_title('Transgender Male Models by Skintone')
    ax[4].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[4].set_ylabel('Count')
    ax[4].set_xticks(list(range(1, 11)))
    ax[4].set_xticklabels(list(range(1, 11)), rotation=90)
    #plt.subplots_adjust(wspace=0.5)
    plt.savefig('images/skintone_vs_gender.png')
    plt.tight_layout(h_pad=100.0)
    plt.show()



def main(): 
    
    skintones_df = import_skintone_data()
    #plot_skintone_distribution()
    #plot_skintone_vs_achievements()

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
    
    #moty_winners_by_skintone(moty_df_model_of_the_year, skintones_df)
    skintone_vs_gender(skintones_df, moty_df, top_50_male, top_50_female)
    
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

    #TODO: top 50 - male'''

    





main()


