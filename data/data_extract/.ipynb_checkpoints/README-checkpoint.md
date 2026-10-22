# Electride screening — feature dataset

CIF → labeled CSV using **five individual tools**: `structure_summary`, `electron_count_hypotheses`,
`bonding_environment`, `cavity_geometry`, `void_connectivity`. Each resulting column is `function__feature`.

Place `electride_features.py`, `labels.json` and `structures/` next to this notebook. No LLM is involved;
labels are supplied, never inferred from the CIFs.

Install if necessary: `%pip install pymatgen ase scipy numpy pandas tqdm`.