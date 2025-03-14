import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    return data

#Ho: We hypothesize that gender and skin tone has no impact on the number of runways, magazine covers, and awards that models earn. 
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


def skintone_vs_runway(skintones_df, moty_df, top_50_f, top_50_m, agency_df):
    combined_data = pd.merge(moty_df, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty', 'current_agency_affiliation':'current_agency'})
    #print('moty:', combined_data.columns)
    combined_data_f = pd.merge(top_50_f, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_f', 'hair ': 'hair_color', 'eyes ': 'eye_color'})
    #print('top_50_f:', combined_data_f.columns)
    combined_data_m = pd.merge(top_50_m, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_m', 'hair ': 'hair_color', 'eyes ': 'eye_color'})
    #print('top_50_m:', combined_data_m.columns)
    combined_data = pd.concat([combined_data, combined_data_f, combined_data_m], axis=0).drop(columns=['agency_city', 'agency_country'])
    agency_df = agency_df.drop(columns=['female', 'male', 'non_binary', 'city', 'country']).rename(columns={'agency_name': 'current_agency'})
    combined_data = pd.merge(combined_data, agency_df, on='current_agency', how='left')
    combined_data = combined_data.drop(columns=['skin_tone_moty', 'skin_tone_top_50_f', 'skin_tone_top_50_m'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)

    return combined_data

def analysis(combined_data):
    '''Ho: We hypothesize that gender and skin tone has no impact on the number of runways, magazine covers, and awards that models earn. 
'''
    combined_data = combined_data.dropna(subset=['gender', 'skin_tone'])
    combined_data['gender'] = combined_data['gender'].astype('category')
    combined_data['skin_tone'] = combined_data['skin_tone'].astype('category')

    model_runway_shows = smf.ols('number_of_runway_shows ~ C(gender) + C(skin_tone) + C(gender):C(skin_tone)', data=combined_data).fit()
    anova_table_shows = sm.stats.anova_lm(model_runway_shows, typ=2) 
    print("ANOVA Results using statsmodels:\n", anova_table_shows)
    residuals = model_runway_shows.resid  # Extract residuals from ANOVA model
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()





    

def main(): 
    
    skintones_df = import_skintone_data()
   

    moty_df = load_data('data/moty.csv')
    agency_df = load_data('data/Agency.csv')
    top_50_female = load_data('data/Top_50_F.csv')
    top_50_male = load_data('data/Top_50_M.csv')

    #remove white space
    moty_df['name'] = moty_df['name'].str.strip()
    moty_df['award'] = moty_df['award'].str.strip()
    moty_df['hair_color'] = moty_df['hair_color'].str.lower().str.strip()
    moty_df['eye_color'] = moty_df['eye_color'].str.lower().str.strip()
    moty_df['choice'] = moty_df['choice'].str.strip()
    moty_df['gender'] = moty_df['gender'].str.strip()   

    #remove duplicate names 
    #moty_df = moty_df.drop_duplicates(subset=['name'], keep='first')

    #preprocess

    #Model of the Year
    moty_df_model_of_the_year = moty_df[moty_df['award'] == 'MOTY']

    #break out star
    moty_bs_df = moty_df[moty_df['award'] == 'BS']

    combined_data = skintone_vs_runway(skintones_df, moty_df_model_of_the_year, top_50_female, top_50_male, agency_df)
    print(combined_data.head(4))

    analysis(combined_data)

    
    
  


    #TODO: top 50 - female

    #TODO: top 50 - male'''

main()