from datasets import Dataset, DatasetDict
import json
from Bio import SeqIO

def fasta_to_dataset(aa_file = "NRPSs_mid-1800.fa", label_file = 'labels.json', characters = "ACDEFGHIKLMNPQRSTVWY"):
    label_dict = json.loads(open(label_file).read())
    train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
    samples = {"id": [], "sequence": [], "class": []}
    for record in train_record_aa:
        label = record.description.split('|')[-1]
        id = record.description.split('|')[0]
        if label in label_dict:
            seq = str(record.seq)
            if not set(seq).issubset(characters):
                continue
            cl = label_dict[label]['class']
            samples["id"].append(id)
            samples["sequence"].append(seq)
            samples["class"].append(cl)
    print("There are " + str(len(samples['id'])) + " sequences in the dataset with correct labeling, using the amino acids in the alphabet.")
    # Use a dictionary to remove duplicates and preserve order
    unique_samples = {"id": [], "sequence": [], "class": []}
    seen_ids = set()

    for i, id in enumerate(samples["id"]):
        if id not in seen_ids:
            seen_ids.add(id)
            unique_samples["id"].append(samples["id"][i])
            unique_samples["sequence"].append(samples["sequence"][i])
            unique_samples["class"].append(samples["class"][i])
    samples = unique_samples
    print("There are " + str(len(samples['id'])) + " sequences when removing duplicates. This is the final dataset.")
    max_len = max([len(seq) for seq in samples["sequence"]])
    min_len = min([len(seq) for seq in samples["sequence"]])
    print("Max length sequence is: " + str(max_len))
    print("Min length sequence is: " + str(min_len))

    # Create a dictionary to store the dataset
    dataset = Dataset.from_dict(samples)
    return dataset

if __name__ == "__main__":
    dataset_train = fasta_to_dataset(aa_file = 'testcase_train.fa', label_file = 'labels_test.json')
    dataset_test = fasta_to_dataset(aa_file = 'testcase_test.fa', label_file = 'labels_test.json')
    dataset_val = fasta_to_dataset(aa_file = 'testcase_val.fa', label_file = 'labels_test.json')

    dataset_dict = DatasetDict({'train': dataset_train, 'test': dataset_test, 'val': dataset_val})
    dataset_dict.push_to_hub('test_dataset')