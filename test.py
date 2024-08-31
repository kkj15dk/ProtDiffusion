from Bio import SeqIO
import re

aa_file = open('UniRef50-processed')


train_record_aa_generator = SeqIO.parse(aa_file, "fasta")

for i, record in enumerate(train_record_aa_generator):
    print('id:', record.id)
    print('description:', record.description)
    match = re.search(r'TaxID=(\d+)', record.description)
    if match:
        tax_id = match.group(1)
        print(tax_id)
    break