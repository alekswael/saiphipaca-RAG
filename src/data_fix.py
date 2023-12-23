# Import the necessary libraries
import pandas as pd

# Read the dataframe from the file
df = pd.read_csv('synthetic_data.csv')

# Define the function to edit the text structure
def edit_text_structure(text):
    # Add the initial instruction and context
    edited_text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction: Use the following pieces of context to answer the users question.\n\n" + text
    
    # Replace "Human:" with "### Input:"
    edited_text = edited_text.replace("Human:", "### Input:")
    
    # Replace "Assistant:" with "### Response:"
    edited_text = edited_text.replace("Assistant:", "### Response:")
    
    return edited_text

# Apply the function to the "synthetic_text" column
df['text_synthetic'] = df['text_synthetic'].apply(edit_text_structure)

# Save the updated dataframe to a new file
df.to_csv('synthetic_data.csv', index=False)

# Print "Done!"
print("Done!")
