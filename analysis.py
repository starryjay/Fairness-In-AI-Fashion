import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    return data

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


def skintone_vs_runway(skintones_df, moty_df, top_50_f, top_50_m, agency_df):
    combined_data = pd.merge(moty_df, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_moty', 'current_agency_affiliation':'current_agency'})
    combined_data_f = pd.merge(top_50_f, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_f', 'hair ': 'hair_color', 'eyes ': 'eye_color'})
    combined_data_m = pd.merge(top_50_m, skintones_df, on='name', how='inner').rename(columns={'skin_tone_y': 'skin_tone', 'skin_tone_x': 'skin_tone_top_50_m', 'hair ': 'hair_color', 'eyes ': 'eye_color'})
    combined_data = pd.concat([combined_data, combined_data_f, combined_data_m], axis=0).drop(columns=['agency_city', 'agency_country'])
    agency_df = agency_df.drop(columns=['female', 'male', 'non_binary', 'city', 'country']).rename(columns={'agency_name': 'current_agency'})
    combined_data = pd.merge(combined_data, agency_df, on='current_agency', how='left')
    combined_data = combined_data.drop(columns=['skin_tone_moty', 'skin_tone_top_50_f', 'skin_tone_top_50_m'])
    combined_data.loc[:, 'skin_tone'] = combined_data.loc[:, 'skin_tone'].astype(int)

    return combined_data

def analysis_runway_shows(combined_data):
    '''Ho: We hypothesize that gender and skin tone has no impact on the number of runways, magazine covers, and awards that models earn. 
'''
    combined_data = combined_data.dropna(subset=['number_of_runway_shows', 'gender', 'skin_tone'])
 
    combined_data['gender'] = combined_data['gender'].astype('category')
    combined_data['skin_tone'] = combined_data['skin_tone'].astype('category')

    model_runway_shows = smf.ols('number_of_runway_shows ~ C(gender) + C(skin_tone) + C(gender):C(skin_tone)', data=combined_data).fit()
    anova_table_shows = sm.stats.anova_lm(model_runway_shows, typ=2) 
    anova_table_shows = anova_table_shows.fillna(0)
    print("ANOVA Results using statsmodels [Runway Shows]:\n", anova_table_shows)
    residuals = model_runway_shows.resid
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title("Residuals Distribution Runway Shows")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig('residuals_runway_shows.png')
    plt.show()

def analysis_magazine_covers(combined_data):
     '''Ho: We hypothesize that gender and skin tone has no impact on the number of magazine covers'''
     combined_data = combined_data.dropna(subset=['number_of_covers', 'gender', 'skin_tone'])
     combined_data['gender'] = combined_data['gender'].astype('category')
     combined_data['skin_tone'] = combined_data['skin_tone'].astype('category')

     model_magazine_covers = smf.ols('number_of_covers ~ C(gender) + C(skin_tone) + C(gender):C(skin_tone)', data=combined_data).fit()
     anova_table_covers = sm.stats.anova_lm(model_magazine_covers, typ=2) 
     anova_table_covers = anova_table_covers.fillna(0)
     print("ANOVA Results using statsmodels [ Magazine Covers]:\n", anova_table_covers)
     residuals = model_magazine_covers.resid
     plt.figure(figsize=(8, 5))
     sns.histplot(residuals, bins=20, kde=True)
     plt.title("Residuals Distribution Covers")
     plt.xlabel("Residuals")
     plt.ylabel("Frequency")
     plt.savefig('residuals_magazine_covers.png')
     plt.show()

def anallysis_awards(combined_data):
    '''Ho: We hypothesize that gender and skin tone has no impact on the number of awards'''
    combined_data = combined_data.dropna(subset=['num_achievements', 'gender', 'skin_tone'])

    combined_data['gender'] = combined_data['gender'].astype('category')

    combined_data['skin_tone'] = combined_data['skin_tone'].astype('category')

    model_awards = smf.ols('num_achievements ~ C(gender) + C(skin_tone) + C(gender):C(skin_tone)', data=combined_data).fit()
    anova_table_award = sm.stats.anova_lm(model_awards, typ=2) 
    anova_table_award = anova_table_award.fillna(0)
    print("ANOVA Results using statsmodels [Awards]:\n", anova_table_award)
    residuals = model_awards.resid
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title("Residuals Distribution Awards")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig('residuals_awards.png')
    plt.show()

def plot_interaction(combined_data, outcome):
    """
    Plots the interaction between gender and skin tone for num_achievements
    """

    skin_tones = sorted(combined_data['skin_tone'].unique())
    genders = combined_data['gender'].unique()


    plt.figure(figsize=(10, 6))

  
    for gender in genders:
        means = []
        for tone in skin_tones:

            mean_val = combined_data[
                (combined_data['gender'] == gender) & 
                (combined_data['skin_tone'] == tone)
            ][outcome].mean()
            means.append(mean_val)

        plt.plot(skin_tones, means, marker='o', label=f'{gender}')


    plt.title('Interaction between Gender and Skin Tone')
    plt.xlabel('Skin Tone')
    plt.ylabel('Number of Achievements')
    plt.legend(title='Gender')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def regresseion_analysis(combined_data):
    '''regression analysis while controlling for nationality, interaction effect of gender and skintone'''
    print(combined_data[['gender', 'skin_tone', 'nationality']].value_counts())
    print(combined_data[['number_of_runway_shows', 'number_of_campaigns_in_last_three_years', 'number_of_covers']].describe())

    model = smf.ols('num_achievements ~ C(gender) * C(skin_tone) + nationality', data=combined_data).fit()
    print(model.summary())
   

    
    





    

def main(): 
    
    skintones_df = import_skintone_data()
   

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

    moty_df_model_of_the_year = moty_df[moty_df['award'] == 'MOTY']

    moty_bs_df = moty_df[moty_df['award'] == 'BS']

    combined_data = skintone_vs_runway(skintones_df, moty_df_model_of_the_year, top_50_female, top_50_male, agency_df)
    print(combined_data.head(4))
    print(combined_data.columns)

    analysis_runway_shows(combined_data)
    analysis_magazine_covers(combined_data)
    anallysis_awards(combined_data)
    plot_interaction(combined_data, 'num_achievements')

    regresseion_analysis(combined_data)

main()