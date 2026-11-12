import pandas as pd

# Define the column names for the Blood Transfusion Data Set
original_column_names = [
    'Recency_months',             # R (Recency - months since last donation)
    'Frequency_total_donations',  # F (Frequency - total number of donation)
    'Monetary_total_cc',          # M (Monetary - total blood donated in c.c.)
    'Time_months_since_first',    # T (Time - months since first donation)
    'Donated_March_2007'          # Binary variable (1=donated, 0=not donated) - This is the target variable (class label)
]

# The path to the uploaded .data file for the Blood Transfusion Data Set
input_file = 'transfusion.data' # Assuming the file is named transfusion.data based on common UCI dataset names

# The desired name for the new, modified CSV file
output_file = 'blood_transfusion_modified.csv'

try:
    # Read the .data file using the defined column names
    # header=None indicates that the file does not have a header row
    blood_transfusion_df = pd.read_csv(input_file, header=None, names=original_column_names)

    # Pop the 'Donated_March_2007' column (class label) to move it to the end
    class_label = blood_transfusion_df.pop('Donated_March_2007')
    # Re-insert the 'Donated_March_2007' column at the end of the DataFrame
    blood_transfusion_df['Donated_March_2007'] = class_label

    # Insert a new 'ID' column at the beginning of the DataFrame for unique identification
    # The ID starts from 1 and goes up to the number of rows in the DataFrame
    blood_transfusion_df.insert(0, 'ID', range(1, len(blood_transfusion_df) + 1))

    # Save the modified DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column in the CSV
    blood_transfusion_df.to_csv(output_file, index=False)

    print(f"Successfully created '{output_file}' with the new format.")
    print("\nHere's a preview of your modified CSV file:")
    print(blood_transfusion_df.head())

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure it's in the same directory as the script or provide the full path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

