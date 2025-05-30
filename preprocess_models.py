import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    return data

def eda_moty_hair_eye_color_choice_distribution(moty_df, award):
    '''plot count of hair and eye color based on choice of award - YES IN FINAL, CHANGE COLORS + ADD TITLE'''
    moty_df = moty_df.loc[moty_df['award'] == award]
    eye_color_map = {'brown': 'saddlebrown', 'blue': 'cornflowerblue', 'green': 'forestgreen', 'blue/green': 'darkcyan', 'hazel': 'olive'}
    plt.figure(figsize=(10, 8))
    sns.catplot(
    data=moty_df,
    x="hair_color",
    hue="eye_color",
    col="choice",
    kind="count",
    height=8,
    aspect=0.8,
    palette=eye_color_map,
    legend_out=True)

    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of type of awards based on hair and eye color')
        plt.tight_layout()
        plt.savefig('images/moty_hair_eye_color_choice_distribution.png')
    else:
        plt.suptitle('Breakout Star - Distribution of type of awards based on hair and eye color')
        plt.tight_layout()
        plt.savefig('images/moty_bs_hair_eye_color_choice_distribution.png')

def eda_moty_hair_eye_color_num_achievements_distribution(moty_df, award):
    '''plot median num of achievements based on hair and eye color - YES IN FINAL'''
    moty_df = moty_df.loc[moty_df['award'] == award]
    eye_color_map = {'brown': 'saddlebrown', 'blue': 'cornflowerblue', 'green': 'forestgreen', 'blue/green': 'darkcyan', 'hazel': 'olive'}

    plt.figure(figsize=(16, 8))
    sns.catplot(
     data=moty_df,
     x="hair_color",
     hue="eye_color",
     y="num_achievements",
     kind="bar",
     height=8,
     aspect=1.5, 
     palette=eye_color_map,
     estimator=np.median,
     legend_out=True)
   
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of achievements based on hair and eye color')
        plt.savefig('images/moty_hair_eye_color_achievements_distribution.png')
    else:
        plt.suptitle('Breakout Star - Distribution of median num of achievements based on hair and eye color')
        plt.savefig('images/moty_bs_hair_eye_color_achievements_distribution.png')


def eda_moty_hair_eye_color_num_runway_shows_distribution(moty_df, award):
    '''plot median num of runway shows based on hair and eye color'''
    eye_color_map = {'brown': 'saddlebrown', 'blue': 'cornflowerblue', 'green': 'forestgreen', 'blue/green': 'darkcyan', 'hazel': 'olive'}
    moty_df = moty_df.loc[moty_df['award'] == award]

    plt.figure(figsize=(16, 8))
    sns.catplot(
     data=moty_df,
     x="hair_color",
     hue="eye_color",
     y="number_of_runway_shows",
     kind="bar",
     height=8,
     aspect=1.5, 
     estimator=np.median,
     palette=eye_color_map,
     legend_out=True)
    
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of runway shows based on hair and eye color')
        plt.savefig('images/moty_hair_eye_color_runway_shows_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of median num of runway shows based on hair and eye color')
        plt.savefig('images/moty_bs_hair_eye_color_runway_shows_distribution.png')

def eda_num_runway_shows_per_gender(moty_df, award):
    '''plot median num of runway shows per gender'''
    moty_df = moty_df.loc[moty_df['award'] == award]

    plt.figure(figsize=(16, 8))
    sns.catplot(
     data=moty_df,
     x='gender',
     y="number_of_runway_shows",
     hue='gender',
     kind="bar",
     height=10,
     aspect=0.8, 
     palette='viridis',
     estimator=np.median)
    
    if award == 'MOTY':
        plt.suptitle('Model of the Year - Distribution of median num of runway shows')
        plt.savefig('images/moty_num_runway_shows_gender_distribution.png')

    else:
        plt.suptitle('Breakout Star - Distribution of median num of runway shows')
        plt.savefig('images/moty_bs_number_runway_shows_gender_distribution.png')

def eda_gender_distribution_over_years(moty_df):
    '''plot gender distribution over the years'''
    gender_per_yr = moty_df.groupby(['year', 'award', 'gender']).size().reset_index(name='count')

    color_map = {"M": "blue", "F": "red","NB":"green", "TF":"purple", "TM":"orange"}

    categories_pivot = gender_per_yr.pivot_table(index=['year', 'award'], columns = 'gender', values = 'count').fillna(0)
    categories_pivot.plot(kind='bar', stacked=True, color = [color_map.get(x) for x in categories_pivot.columns], figsize=(15, 10))

    plt.title("Gender Distribution over the years")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend(title = "gender")
    plt.savefig("images/award_won_by_gender_over_years_matplotlib.png")



def gender_distribution_per_us_agency(agency_df):
    '''gender distribution per agency - INCLUDE '''

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
    plt.xticks(rotation = 45, ha= 'right')
    plt.margins(x = 0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
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
    skintones[['R', 'G', 'B']] = skintones['RGB_value'].str.split(',', expand=True).astype(int)
    skintones = skintones.drop(columns=['RGB_value'])
    skintones = skintones.reset_index(drop=True)
    skintones.to_csv('data/skintones.csv', index = False)
    return skintones


def plot_skintone_distribution():
    skintones_df = load_data('data/skintones.csv')
    skintones_df['R'] = skintones_df['R'].astype(int)
    skintones_df['G'] = skintones_df['G'].astype(int)
    skintones_df['B'] = skintones_df['B'].astype(int)

    plt.figure(figsize=(10, 8))
    skintone_counts = skintones_df['skin_tone'].value_counts().sort_index()
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
    combined_data.rename(columns={'skin_tone_x': 'skin_tone', 'skin_tone_y': 'skin_tone_moty'}, inplace=True)
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty'])
    plt.figure(figsize=(10, 8))
    combined_data['skin_tone'] = combined_data['skin_tone'].astype(int)
    mean_achievements = combined_data.groupby('skin_tone')['num_achievements'].mean()
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    plt.bar(x=list(mean_achievements.index), height=list(mean_achievements.values), color=[skin_tone_colormap[i] for i in mean_achievements.index])
    plt.title('Median Number of Achievements per Skintone')
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
    plt.bar(x=list(skintone_counts.index), height=list(skintone_counts.values), color=[skin_tone_colormap[i] for i in skintone_counts.index])
    plt.title('MOTY Winners by Skintone')
    plt.xlabel('Skin Tone on Monk Skin Tone Scale')
    plt.ylabel('Count')
    plt.xticks(list(range(1, 11)), rotation=90)
    plt.savefig('images/moty_winners_by_skintone.png')
    plt.show()

def skintone_vs_gender(skintones_df, moty_df, top_50_male, top_50_female):
    Agency = pd.read_csv('data/Agency.csv')
    moty_df.rename({'current_agency_affiliation': 'agency_name'}, inplace=True, axis=1)
    top_50_female.rename({'current_agency': 'agency_name'}, inplace=True, axis=1)
    top_50_male.rename({'current_agency': 'agency_name'}, inplace=True, axis=1)
    models = pd.concat([moty_df, top_50_female, top_50_male], axis=0)
    models = pd.merge(left=models, right=Agency, on='agency_name', how='outer')
    models = models.loc[models['agency_country'] == 'USA']
    models = models.loc[:, ['name', 'gender']].drop_duplicates().reset_index(drop = True)
    combined_data = pd.merge(skintones_df, models, on='name', how='inner')
    combined_data = combined_data.drop(columns=['R', 'G', 'B'])
    combined_data['skin_tone'] = combined_data['skin_tone'].astype(int)
    grouped_data = combined_data.groupby('gender')['skin_tone'].value_counts().unstack().fillna(0)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    if "TM" in grouped_data.index:
        fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    else:
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    ax[0].bar(x=grouped_data.columns, height=grouped_data.iloc[0], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[0].set_title('Cisgender Female Models')
    ax[0].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(list(range(1, 11)))
    ax[0].set_xticklabels(list(range(1, 11)))
    ax[0].set_yticks(list(range(0, 21, 2)))
    ax[0].set_yticklabels(list(range(0, 21, 2)))
    ax[1].bar(x=grouped_data.columns, height=grouped_data.iloc[1], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[1].set_title('Cisgender Male Models')
    ax[1].set_xlabel('Skin Tone')
    ax[1].set_xticks(list(range(1, 11)))
    ax[1].set_xticklabels(list(range(1, 11)))
    ax[1].set_yticks(list(range(0, 21, 2)))
    ax[1].set_yticklabels(list(range(0, 21, 2)))
    ax[2].bar(x=grouped_data.columns, height=grouped_data.iloc[2], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[2].set_title('Non-Binary Models')
    ax[2].set_xlabel('Skin Tone')
    ax[2].set_xticks(list(range(1, 11)))
    ax[2].set_xticklabels(list(range(1, 11)))
    ax[2].set_yticks(list(range(0, 21, 2)))
    ax[2].set_yticklabels(list(range(0, 21, 2)))
    ax[3].bar(x=grouped_data.columns, height=grouped_data.iloc[3], color=[skin_tone_colormap[i] for i in grouped_data.columns])
    ax[3].set_title('Transgender Female Models')
    ax[3].set_xlabel('Skin Tone')
    ax[3].set_xticks(list(range(1, 11)))
    ax[3].set_xticklabels(list(range(1, 11)))
    ax[3].set_yticks(list(range(0, 21, 2)))
    ax[3].set_yticklabels(list(range(0, 21, 2)))
    if "TM" in grouped_data.index:
        ax[4].bar(x=grouped_data.columns, height=grouped_data.iloc[4], color=[skin_tone_colormap[i] for i in grouped_data.columns])
        ax[4].set_title('Transgender Male Models')
        ax[4].set_xticks(list(range(1, 11)))
        ax[4].set_xticklabels(list(range(1, 11)))
        ax[4].set_yticks(list(range(0, 21, 2)))
        ax[4].set_yticklabels(list(range(0, 21, 2)))
    plt.suptitle('Skintone Distribution By Gender for All USA Models')
    plt.savefig('images/skintone_vs_gender.png')
    plt.show()

def skintone_vs_moty_bs(skintones_df, moty_df):
    plt.figure(figsize=(10, 8))
    combined_data = pd.merge(moty_df, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty'})
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    skintone_counts_moty = combined_data.loc[combined_data['award'] == 'MOTY'].loc[:, 'skin_tone'].value_counts().sort_index()
    skintone_counts_bs = combined_data.loc[combined_data['award'] == 'BS'].loc[:, 'skin_tone'].value_counts().sort_index()
    skintone_counts_moty = skintone_counts_moty.reindex(list(range(1, 11)), fill_value=0)
    skintone_counts_bs = skintone_counts_bs.reindex(list(range(1, 11)), fill_value=0)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
    ax[0].bar(x=list(skintone_counts_moty.index), height=list(skintone_counts_moty.values), color=[skin_tone_colormap[i] for i in skintone_counts_moty.index])
    ax[0].set_title('MOTY Winners by Skintone')
    ax[0].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(list(range(1, 11)))
    ax[0].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[0].set_yticks(list(range(0, 21, 2)))
    ax[0].set_yticklabels(list(range(0, 21, 2)))
    ax[1].bar(x=list(skintone_counts_bs.index), height=list(skintone_counts_bs.values), color=[skin_tone_colormap[i] for i in skintone_counts_bs.index])
    ax[1].set_title('Breakout Star Winners by Skintone')
    ax[1].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[1].set_ylabel('Count')
    ax[1].set_xticks(list(range(1, 11)))
    ax[1].set_xticklabels(list(range(1, 11)), rotation=90)
    ax[1].set_yticks(list(range(0, 21, 2)))
    ax[1].set_yticklabels(list(range(0, 21, 2)))
    plt.suptitle('MOTY and Breakout Star Winners by Skintone')
    plt.savefig('images/moty_bs_winners_by_skintone.png')
    plt.show()
    
def skintone_vs_runway(skintones_df, moty_df, top_50_f, top_50_m):
    plt.figure(figsize=(10, 8))
    combined_data = pd.merge(moty_df, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty'})
    combined_data_f = pd.merge(top_50_f, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_f'})
    combined_data_m = pd.merge(top_50_m, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_m'})
    combined_data = pd.concat([combined_data, combined_data_f, combined_data_m], axis=0)
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty', 'skin_tone_top_50_f', 'skin_tone_top_50_m'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    fig, axs = plt.subplots(3, 3, figsize=(20, 20), sharey=True, sharex=True)
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    fig, axs = plt.subplots(3, 3, figsize=(20, 20), sharey=True, sharex=True)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96', 
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    axs[0, 0].hist(combined_data.loc[combined_data['skin_tone'] == 1, 'number_of_runway_shows'], color=skin_tone_colormap[1])
    axs[0, 0].set_title('Monk 01')
    axs[0, 1].hist(combined_data.loc[combined_data['skin_tone'] == 2, 'number_of_runway_shows'], color=skin_tone_colormap[2])
    axs[0, 1].set_title('Monk 02')
    axs[0, 2].hist(combined_data.loc[combined_data['skin_tone'] == 3, 'number_of_runway_shows'], color=skin_tone_colormap[3])
    axs[0, 2].set_title('Monk 03')
    axs[1, 0].hist(combined_data.loc[combined_data['skin_tone'] == 4, 'number_of_runway_shows'], color=skin_tone_colormap[4])
    axs[1, 0].set_title('Monk 04')
    axs[1, 1].hist(combined_data.loc[combined_data['skin_tone'] == 5, 'number_of_runway_shows'], color=skin_tone_colormap[5])
    axs[1, 1].set_title('Monk 05')
    axs[1, 2].hist(combined_data.loc[combined_data['skin_tone'] == 6, 'number_of_runway_shows'], color=skin_tone_colormap[6])
    axs[1, 2].set_title('Monk 06')
    axs[2, 0].hist(combined_data.loc[combined_data['skin_tone'] == 7, 'number_of_runway_shows'], color=skin_tone_colormap[7])
    axs[2, 0].set_title('Monk 07')
    axs[2, 1].hist(combined_data.loc[combined_data['skin_tone'] == 8, 'number_of_runway_shows'], color=skin_tone_colormap[8])
    axs[2, 1].set_title('Monk 08')
    axs[2, 2].hist(combined_data.loc[combined_data['skin_tone'] == 9, 'number_of_runway_shows'], color=skin_tone_colormap[9])
    axs[2, 2].set_title('Monk 09')
    plt.suptitle('Number of Runway Shows by Skintone')
    plt.savefig('images/runway_shows_by_skintone.png')
    plt.show()

def mean_skintone_of_moty_bs_over_time(skintone_df, moty_df):
    moty_df['year'] = pd.to_datetime(moty_df['year'], format='%Y')
    combined_data = pd.merge(moty_df, skintone_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty'})
    combined_data = combined_data.drop(columns=['R', 'G', 'B', 'skin_tone_moty'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    mean_skintone_by_year = combined_data.groupby('year')['skin_tone'].mean()
    plt.figure(figsize=(10, 8))
    plt.plot(mean_skintone_by_year.index, mean_skintone_by_year.values)
    plt.title('Mean Skintone of Model of the Year and Breakout Star Winners in the last 10 years')
    plt.xlabel('Year')
    plt.ylabel('Mean Skintone on Monk Skin Tone Scale')
    plt.savefig('images/mean_skintone_of_moty_bs_over_time.png')
    plt.show()

def campaigns_vs_skintone(top_50_female, top_50_male, skintones_df):
    top_50 = pd.concat([top_50_female, top_50_male], axis=0).rename(columns={'current_agency': 'agency_name'})
    Agency = pd.read_csv('data/Agency.csv')
    top_50 = pd.merge(left=top_50, right=Agency, on='agency_name', how='outer')
    top_50 = top_50.loc[top_50['agency_country'] == 'USA']
    top_50 = top_50.loc[:, ['name', 'number_of_campaigns_in_last_three_years', 'gender']].drop_duplicates().reset_index(drop = True)
    combined_data = pd.merge(skintones_df, top_50, on='name', how='inner').rename(columns={'skin_tone_x': 'skin_tone', 'skin_tone_y': 'skin_tone_top_50'})
    combined_data = combined_data.drop(columns=['R', 'G', 'B'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96',
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    sns.boxplot(ax=ax[0], data=pd.concat([combined_data.loc[combined_data['gender'] == 'F'], 
                                          combined_data.loc[combined_data['gender'] == 'TF'], 
                                          combined_data.loc[combined_data['gender'] == 'NB']], axis=0), 
                                          x='skin_tone', y='number_of_campaigns_in_last_three_years', hue='skin_tone',
                                          palette=skin_tone_colormap, saturation=1, linecolor='xkcd:burgundy', legend=False)
    ax[0].set_title('Top 50 Female Models')
    ax[0].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[0].set_ylabel('Number of Campaigns in Last Three Years')
    sns.boxplot(ax=ax[1], data=pd.concat([combined_data.loc[combined_data['gender'] == 'M'], 
                                          combined_data.loc[combined_data['gender'] == 'TM'], 
                                          combined_data.loc[combined_data['gender'] == 'NB']], axis=0), 
                                          x='skin_tone', y='number_of_campaigns_in_last_three_years', hue='skin_tone',
                                          palette=skin_tone_colormap, saturation=1, linecolor='midnightblue', legend=False)
    ax[1].set_title('Top 50 Male Models')
    ax[1].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[1].set_ylabel('Number of Campaigns in Last Three Years')
    plt.suptitle('Number of Campaigns vs Skintone for Top 50 Models')
    plt.savefig('images/campaigns_vs_skintone.png')
    plt.show()

def covers_vs_skintone(top_50_female, top_50_male, skintones_df):
    top_50 = pd.concat([top_50_female, top_50_male], axis=0).rename(columns={'current_agency': 'agency_name'})
    Agency = pd.read_csv('data/Agency.csv')
    top_50 = pd.merge(left=top_50, right=Agency, on='agency_name', how='outer')
    top_50 = top_50.loc[top_50['agency_country'] == 'USA']
    top_50 = top_50.loc[:, ['name', 'number_of_covers', 'gender']].drop_duplicates().reset_index(drop = True)
    combined_data = pd.merge(skintones_df, top_50, on='name', how='inner').rename(columns={'skin_tone_x': 'skin_tone', 'skin_tone_y': 'skin_tone_top_50'})
    combined_data = combined_data.drop(columns=['R', 'G', 'B'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    skin_tone_colormap = {1: '#f6ede4', 2: '#f3e7db', 3: '#f7ead0', 4: '#eadaba', 5: '#d7bd96',
                          6: '#a07e56', 7: '#825c43', 8: '#604134', 9: '#3a312a', 10: '#292420'}
    sns.boxplot(ax=ax[0], data=pd.concat([combined_data.loc[combined_data['gender'] == 'F'], 
                                          combined_data.loc[combined_data['gender'] == 'TF'], 
                                          combined_data.loc[combined_data['gender'] == 'NB']], axis=0), 
                                          x='skin_tone', y='number_of_covers', hue='skin_tone',
                                          palette=skin_tone_colormap, saturation=1, linecolor='xkcd:burgundy', legend=False)
    ax[0].set_title('Top 50 Female Models')
    ax[0].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[0].set_ylabel('Number of Magazine Covers in Last Three Years')
    sns.boxplot(ax=ax[1], data=pd.concat([combined_data.loc[combined_data['gender'] == 'M'], 
                                          combined_data.loc[combined_data['gender'] == 'TM'], 
                                          combined_data.loc[combined_data['gender'] == 'NB']], axis=0), 
                                          x='skin_tone', y='number_of_covers', hue='skin_tone',
                                          palette=skin_tone_colormap, saturation=1, linecolor='midnightblue', legend=False)
    ax[1].set_title('Top 50 Male Models')
    ax[1].set_xlabel('Skin Tone on Monk Skin Tone Scale')
    ax[1].set_ylabel('Number of Magazine Covers in Last Three Years')
    plt.suptitle('Number of Magazine Covers vs Skintone for Top 50 Models')
    plt.savefig('images/covers_vs_skintone.png')
    plt.show()


def main(): 
    
    skintones_df = import_skintone_data()
    plot_skintone_distribution()
    plot_skintone_vs_achievements()

    moty_df = load_data('data/moty.csv')
    agency_df = load_data('data/Agency.csv')
    top_50_female = load_data('data/Top_50_F.csv')
    top_50_male = load_data('data/Top_50_M.csv')

    moty_df['name'] = moty_df['name'].str.strip()
    moty_df['award'] = moty_df['award'].str.strip()
    moty_df['hair_color'] = moty_df['hair_color'].str.lower().str.strip()
    moty_df['eye_color'] = moty_df['eye_color'].str.lower().str.strip()
    moty_df['choice'] = moty_df['choice'].str.strip()
    moty_df['gender'] = moty_df['gender'].str.strip()   

    moty_df = moty_df.drop_duplicates(subset=['name'], keep='first')

    moty_df_model_of_the_year = moty_df[moty_df['award'] == 'MOTY']

    moty_bs_df = moty_df[moty_df['award'] == 'BS']
    
    skintone_vs_gender(skintones_df, moty_df, top_50_male, top_50_female)
    skintone_vs_moty_bs(skintones_df, moty_df)
    skintone_vs_runway(skintones_df, moty_df, top_50_female, top_50_male)
    mean_skintone_of_moty_bs_over_time(skintones_df, moty_df)
    campaigns_vs_skintone(top_50_female, top_50_male, skintones_df)
    covers_vs_skintone(top_50_female, top_50_male, skintones_df)
    
    print("Distribution of type of awards based on hair and eye color")
    
    print(moty_df['hair_color'].value_counts())
    print(moty_df['eye_color'].value_counts())
    print(moty_df['choice'].value_counts())

    eda_moty_hair_eye_color_choice_distribution(moty_df_model_of_the_year, 'MOTY')
    eda_moty_hair_eye_color_num_achievements_distribution(moty_df_model_of_the_year, 'MOTY')

    eda_moty_hair_eye_color_choice_distribution(moty_bs_df, 'BS')
    eda_moty_hair_eye_color_num_achievements_distribution(moty_bs_df, 'BS')

    eda_moty_hair_eye_color_num_runway_shows_distribution(moty_df_model_of_the_year, 'MOTY')
    eda_moty_hair_eye_color_num_runway_shows_distribution(moty_bs_df, 'BS')

    eda_num_runway_shows_per_gender(moty_df_model_of_the_year, 'MOTY')
    eda_num_runway_shows_per_gender(moty_bs_df, 'BS')

    eda_gender_distribution_over_years(moty_df)

    gender_distribution_per_us_agency(agency_df)

main()


