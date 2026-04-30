import pandas as pd

# File paths
predictions_path = "plasmodium_3d7_oof_predictions_with_sequences.csv"
plasmofab_path = "PlasmoFAB_seq.csv"
output_path = "plasmodium_3d7_oof_compared_to_plasmofab.csv"

# 1. Load the data
print("Loading datasets...")
df_pred = pd.read_csv(predictions_path)
# Reading PlasmoFAB based on your 'head' output
df_fab = pd.read_csv(plasmofab_path)

# 2. Define labels based on your provided format (0=Non-Antigen, 1=Antigen)
def map_label(val):
    try:
        # Convert to string and take the first character to handle potential float/formatting issues
        v = str(val).strip()
        if v == '1':
            return "Antigen"
        elif v == '0':
            return "Non-Antigen"
    except:
        pass
    return "Unknown"

# 3. Standardize sequences for exact matching
# Based on your file, the sequence column is named 'seq'
print("Preparing sequence lookup...")
df_fab['seq'] = df_fab['seq'].astype(str).str.strip().str.upper()

# Map the labels using the third column (Unnamed: 2 or however pandas read the empty header)
# We will identify the label column dynamically based on your data structure
label_col = df_fab.columns[2] 
label_lookup = dict(zip(df_fab['seq'], df_fab[label_col].apply(map_label)))

# 4. Perform Comparison
print("Matching 3D7 proteome against PlasmoFAB...")
df_pred['sequence_upper'] = df_pred['sequence'].astype(str).str.strip().str.upper()
df_pred['plasmofab_label'] = df_pred['sequence_upper'].map(label_lookup).fillna("Not Found")

# 5. Save and Clean up
df_pred = df_pred.drop(columns=['sequence_upper'])
df_pred.to_csv(output_path, index=False)

print(f"Success! Comparison saved to: {output_path}")

# Summary statistics for your Methodology section
total = len(df_pred)
found = len(df_pred[df_pred['plasmofab_label'] != "Not Found"])
print(f"\nAnalysis Summary:")
print(f"- Total Proteome Sequences: {total}")
print(f"- Sequences existing in PlasmoFAB: {found}")
print(f"- Novel Candidates identified: {total - found}")
