import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set the data path
data_path = "results/cosine_similarity/"

# First make a barplot for the joint datasets, make a list of the csv files to be used - all .csv files in the data_path folder that start with "joint"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f.startswith('joint')]

# Now we need to extract the "cosine_similarity" column from each of the csv files and append them to a dictionary, then make a dataframe from the dictionary and plot it
cosine_similarity_dict_1 = {}

# Initialize a loop per each csv file
for csv_file in csv_files:

    # Load the data
    df_1 = pd.read_csv(data_path + csv_file)

    # Add the "cosine_similarity" column to the dictionary
    # The key is the csv_file name without the .csv extension and without the "joint_paper_results_" part
    csv_file = csv_file[20:]
    csv_file = csv_file[:-4]
    cosine_similarity_dict_1[csv_file] = df_1['cosine_similarity']

# Now make a dataframe from the dictionary
df_1 = pd.DataFrame.from_dict(cosine_similarity_dict_1)

# Now to do it for the datasets that start with "single"
# Make a list of the csv files to be used - all .csv files in the data_path folder that start with "single"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f.startswith('single')]

# Now we need to extract the "cosine_similarity" column from each of the csv files and append them to a dictionary, then make a dataframe from the dictionary and plot it
cosine_similarity_dict_2 = {}

# Initialize a loop per each csv file
for csv_file in csv_files:

    # Load the data
    df_2 = pd.read_csv(data_path + csv_file)

    # Add the "cosine_similarity" column to the dictionary
    # The key is the csv_file name without the .csv extension and without the "single_paper_results_" part
    csv_file = csv_file[21:]
    csv_file = csv_file[:-4]
    cosine_similarity_dict_2[csv_file] = df_2['cosine_similarity']

# Now make a dataframe from the dictionary
df_2 = pd.DataFrame.from_dict(cosine_similarity_dict_2)

# Now plot the dataframe
fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 5))

category_order = ['phi-1_5', 'phipaca', 'saiphipaca']
color_palette = {'phi-1_5': 'pink', 'phipaca': 'olive', 'saiphipaca': 'cyan'}

sns.boxplot(ax=axes[0], data=df_2, palette=color_palette, order=category_order)
sns.boxplot(ax=axes[1], data=df_1, palette=color_palette, order=category_order)
axes[0].set(ylabel = "Cosine similarity", title = "Single PDF task")
axes[1].set(title = "Joint PDF task")
fig.suptitle("Cosine similarity with gpt-3.5-turbo output by task")

# Save the plot
plt.savefig(data_path + "cosine_similarity.png")