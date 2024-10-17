import os
import pandas as pd

"""
This script converts .txt files containing stock price data into .csv format, adds predefined headers,
and saves the files in a specified output folder.

Usage:
- Ensure the input folder has .txt files in the expected format.
- The output folder will be automatically created if it doesn't exist.

Dependencies:
- pandas
"""


# Set the base directory and subdirectories
base_directory = r'C:\Users\johns\Documents\Masters\Diss\Data'
input_folder = os.path.join(base_directory, 'TXTFormat')
output_folder = os.path.join(base_directory, 'CSVFolder')

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the column headers
headers = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Loop through all .txt files in the input directory
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # Read the file
        file_path = os.path.join(input_folder, filename)
        data = pd.read_csv(file_path, header=None)
        
        # Add the header row
        data.columns = headers
        
        # Save the DataFrame to a CSV file in the output folder
        output_filename = os.path.join(output_folder, filename.replace('.txt', '.csv'))
        data.to_csv(output_filename, index=False)

print("Conversion from .txt to .csv completed.")




