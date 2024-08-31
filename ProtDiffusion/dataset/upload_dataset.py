from datasets import Dataset, DatasetDict
import json
from Bio import SeqIO
from typing import Optional, Tuple
import re

def fasta_to_dataset(aa_file: str, characters = "ACDEFGHIKLMNOPQRSTUVWY") -> Dataset:
    train_record_aa_generator = SeqIO.parse(aa_file, "fasta")
    samples = {"id": [], "sequence": [], "TaxID": [], "RepID": []}
    for record in train_record_aa_generator:
        match = re.search(r'TaxID=(\d+)', record.description)
        if match:
            tax_id = match.group(1)
        else:
            tax_id = None
        
        match = re.search(r'RepID=([\w_]+)', record.description)
        if match:
            rep_id = match.group(1)
        else:
            print("RepID not found for record: " + record.id)
            print("Description: " + record.description)
            continue # skip this record if the tax id is not found
        
        seq = str(record.seq)
        if not set(seq).issubset(characters):
            print("Sequence contains characters not in the alphabet for " + record.id + " : " + seq)
            print("Characters not in the alphabet: " + str(set(seq) - set(characters)))
            continue # skip this record if it contains characters not in the alphabet

        samples["id"].append(record.id)
        samples["sequence"].append(seq)
        samples["TaxID"].append(tax_id)
        samples["RepID"].append(rep_id)
    print("There are " + str(len(samples['id'])) + " sequences in the dataset with correct labeling, using the amino acids in the alphabet.")

    # Create a dictionary to store the dataset
    dataset = Dataset.from_dict(samples)
    return dataset

if __name__ == "__main__":
    dataset = fasta_to_dataset(aa_file = 'UniRef50-processed.faa')

    dataset_dict = DatasetDict({'train': dataset})
    dataset_dict.push_to_hub('UniRef90-preprocessed')