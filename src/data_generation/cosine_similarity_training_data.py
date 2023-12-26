import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import random

# Don't print out warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(420)

data_path = "data/SAI_dataset/"

# Load the data
df = pd.read_csv('SAI_dataset_0312.csv')

# Convert all 'text' values to string
df['text'] = df['text'].astype(str)

# Create a dataframe which only includes pages that are within the defined range 2<page<9
# This is done to prevent comparing "noisy" pages such as the front pages or reference lists which might contain information about the authors/references/other, which might be ignored by GPT4 as it is prompted to specifically extract paragraphs related to the research article.
df = df[df['page'].between(3,9)]

cosine_similarity_list = []

# Make a dataframe for storing text 
df_text = pd.DataFrame(columns = ['pdf_name', 'page', 'gpt_text', 'pdf_text', 'cosine_similarity', 'text_length_diff'])

# Initialize a loop per each pdf_name
for pdf_name in df['pdf_name'].unique():

	# Make a temporary dataframe
	df_temp = df[df['pdf_name'] == pdf_name]

	for unique_page in df_temp['page'].unique():

	    # Concatenate all 'text' values into one string
	    text = ' '.join(df_temp[df_temp['page'] == unique_page]['text'].values)

	    # Now use PyPDF to extract text from the pdf_name
	    pdfFileObj = open(data_path + pdf_name + ".pdf", 'rb')
	    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
	    page_num = unique_page - 1
	    page = pdfReader.getPage(page_num)
	    pdf_text = page.extractText()

	    # Calculate the difference in length between the two texts
	    text_length_diff = abs(len(text) - len(pdf_text))

	    # Now we have the text from the pdf_name and the text from the dataframe
	    # Now we need to compare the two using cosine similarity
	    # First, we need to vectorize the two texts
	    vectorizer = TfidfVectorizer()

	    # Create a list of the two texts
	    corpus = [text, pdf_text]

	    # Now vectorize the corpus
	    X = vectorizer.fit_transform(corpus)

	    # Compute the cosine similarity between the two texts
	    cos_sim = cosine_similarity(X[0], X[1])[0][0]

	    # Append pdf_name, page, gpt_text, pdf_text, cosine_similarity to df_paragraphs
		df_text = df_text.append({'pdf_name': pdf_name, 'page': page_num, 'gpt_text': text, 'pdf_text': pdf_text, 'cosine_similarity' : cos_sim, 'text_length_diff' : text_length_diff}, ignore_index = True)

print("Cosine similarity mean for all pages is: " + df_text['cosine_similarity'].mean())