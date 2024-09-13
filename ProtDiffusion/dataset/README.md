## How the datasets are made
1.
Using the SPARQL query in UniRefALL_SPARQLquery, a csv file is made. It contains all UniProtKB protein sequences (reviewed and unreviewed), along with the kingdom (Eukaryotic = 2759, Prokaryotic = 2). Only Prokaryotic and Eukaryotic sequences are included. They are filtered to not include Non-terminal residues, and Non-adjacent residues, to ignore fragments.

2.
Sort the file using UNIX sort:
(head -n 1 <inputfile.csv> && tail -n +2 <inputfile.csv> | sort -u) > <inputfile-sorted.csv>

This sorts the file, ignoring the header, and deletes duplicates.

3.
The dataset is then encoded using the script convert_csv_to_dataset.py. This script both creates an encoded dataset, and then afterwards creates a grouped dataset, grouped by the clusterid. They are saved to disk as huggingface Datasets.
The grouped dataset is the final one used for the model.