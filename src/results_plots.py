import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set the data path
data_path = "results/cosine_similarity/"

# First make a barplot for the joint datasets, make a list of the csv files to be used - all .csv files in the data_path folder that start with "joint"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f.startswith('joint')]

# Now we need to extract the "cosine_similarity" column from each of the csv files and append them to a dictionary, then make a dataframe from the dictionary and plot it
cosine_similarity_dict = {}

# Initialize a loop per each csv file
for csv_file in csv_files:

    # Load the data
    df = pd.read_csv(data_path + csv_file)

    # Add the "cosine_similarity" column to the dictionary
    # The key is the csv_file name without the .csv extension and without the "joint_paper_results_" part
    csv_file = csv_file[20:]
    csv_file = csv_file[:-4]
    cosine_similarity_dict[csv_file] = df['cosine_similarity']

# Now make a dataframe from the dictionary
df = pd.DataFrame.from_dict(cosine_similarity_dict)

# Now plot the dataframe
sns.boxplot(data=df)
plt.title("Cosine similarity for the joint PDF datasets")

# Save the plot
plt.savefig(data_path + "joint_cosine_similarity.png")

# Clear the plot
plt.clf()

# Now to do it for the datasets that start with "single"
# Make a list of the csv files to be used - all .csv files in the data_path folder that start with "single"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f.startswith('single')]

# Now we need to extract the "cosine_similarity" column from each of the csv files and append them to a dictionary, then make a dataframe from the dictionary and plot it
cosine_similarity_dict = {}

# Initialize a loop per each csv file
for csv_file in csv_files:

    # Load the data
    df = pd.read_csv(data_path + csv_file)

    # Add the "cosine_similarity" column to the dictionary
    # The key is the csv_file name without the .csv extension and without the "single_paper_results_" part
    csv_file = csv_file[21:]
    csv_file = csv_file[:-4]
    cosine_similarity_dict[csv_file] = df['cosine_similarity']

# Now make a dataframe from the dictionary
df = pd.DataFrame.from_dict(cosine_similarity_dict)

# Now plot the dataframe
sns.boxplot(data=df)
plt.title("Cosine similarity for the single PDF datasets")

# Save the plot
plt.savefig(data_path + "single_cosine_similarity.png")