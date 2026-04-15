import pandas as pd
from pathlib import Path

# Set paths based on the script's current location
current_dir = Path(__file__).parent
input_path = current_dir / '2024-12-SGFTFN_ORTHOPYROXENES.csv'
output_path = current_dir / 'natural_opx_cleaned.csv'

print(f"Reading data from: {input_path}")
print("Processing... this may take a moment.")

# Load raw file (latin1 handles special characters in author names/locations)
df = pd.read_csv(input_path, low_memory=False, encoding='latin1')

# Filter exclusively for orthopyroxene rows
df_opx = df[df['MINERAL'].astype(str).str.contains('ORTHOPYROXENE', case=False, na=False)]

# Define columns to extract based on the exact GEOROC header format
cols_to_keep = [
    'CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LOCATION',
    'SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'CR2O3(WT%)', 
    'FEOT(WT%)', 'MNO(WT%)', 'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)'
]

# Ensure columns exist to prevent crashes
available_cols = [c for c in cols_to_keep if c in df_opx.columns]
df_clean = df_opx[available_cols].copy()

# Rename columns to perfectly match Phase 3R ML training features
# PATCH 1: 'FEOT(WT%)' must map to 'FeO_total' for nb08
rename_dict = {
    'SIO2(WT%)': 'SiO2', 'TIO2(WT%)': 'TiO2', 'AL2O3(WT%)': 'Al2O3',
    'CR2O3(WT%)': 'Cr2O3', 'FEOT(WT%)': 'FeO_total', 'MNO(WT%)': 'MnO',
    'MGO(WT%)': 'MgO', 'CAO(WT%)': 'CaO', 'NA2O(WT%)': 'Na2O'
}
df_clean = df_clean.rename(columns=rename_dict)

# Identify just the numeric oxide columns
target_cols = [rename_dict[k] for k in rename_dict if k in available_cols]

# PATCH 2: Force all oxide columns to floats to wipe out strings like '<0.01'
for col in target_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# PATCH 3: Clip negative microprobe artifacts to 0.0001 to prevent ALR log() errors
df_clean[target_cols] = df_clean[target_cols].clip(lower=0.0001)

# Drop rows missing any of the required major oxides
df_clean = df_clean.dropna(subset=target_cols)

# Save the lightweight, model-ready file
df_clean.to_csv(output_path, index=False)
print(f"Success. Saved {len(df_clean)} model-ready OPX rows to: {output_path}")