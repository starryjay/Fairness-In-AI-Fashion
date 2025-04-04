import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def slide_vs_mst(df: pd.DataFrame) -> None:
    """
    Create a plot comparing slide number on the Instagram post to Monk Skin Tone of the AI-generated model.
    Args:
        df: DataFrame containing the data.
    """
    plt.figure(figsize=(4, 4), dpi=300)
    sns.boxplot(x='slide', y='mst', data=df, palette='Set2')
    plt.title('Slide Number vs. Monk Skin Tone')
    plt.xlabel('Slide Number')
    plt.ylabel('Monk Skin Tone')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('images/slide_vs_mst.png')
    plt.show()

def num_ethnicity(df: pd.DataFrame) -> None:
    """
    Plot the frequency of each ethnicity in the dataset.
    Args:
        df: DataFrame containing the data.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(df['observed_ethnicity'].value_counts().index, df['observed_ethnicity'].value_counts().values, color=['#96dce6', '#70bdc8', '#3a8e9a', '#24717c'], width=0.4)
    plt.title('Observed Ethnicity Frequency')
    plt.xlabel('Ethnicity Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/eth_freq.png')
    plt.show()

def main() -> None:
    # Load the data
    copymag_data = pd.read_csv('data/copy_mag_data.csv')
    print(copymag_data[copymag_data['mst'] == 'na'])
    copymag_no_na = copymag_data.drop(copymag_data[copymag_data['mst'] == 'na'].index)
    copymag_no_na['slide'] = copymag_no_na['slide'].astype(int)
    copymag_no_na['mst'] = copymag_no_na['mst'].astype(int)
    slide_vs_mst(copymag_no_na)
    # replace 'na' value in 'observed_ethnicity' column with 'ambiguous'
    copymag_data.loc[:, 'observed_ethnicity'].replace('na', 'ambiguous', inplace=True)
    num_ethnicity(copymag_data)

if __name__ == "__main__":
    main()