from dataclasses import dataclass

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split

from transformers import PreTrainedTokenizerFast


@dataclass
class TrainingConfig:
    padding_side = "right"  # where to put the padding tokens.
    output_dir = "protein_tokenizer"  # the model name locally and on the HF Hub
    pad_token = "-"
    unk_token = "?"
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    repo_id = "kkj15dk/protein_tokenizer"  # the name of the repository to create on the HF Hub

config = TrainingConfig()

# tokenizer for the decoder
tokenizer = Tokenizer(WordLevel(unk_token=config.unk_token))
tokenizer.pre_tokenizer = Split("", behavior='removed')
tokenizer.decoder = decoders.Fuse()

trainer = WordLevelTrainer(special_tokens=[config.pad_token, config.unk_token])

files = ['characters.txt']
tokenizer.train(files, trainer)

# tokenizer for the encoder
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                         pad_token=config.pad_token,
                                         unk_token=config.unk_token,
                                         padding_side=config.padding_side,
                                         clean_up_tokenization_spaces=False,
)

fast_tokenizer.save_pretrained('tokenizer_v4.2')

def encode(example):
    return fast_tokenizer(example['sequence'],
                    padding = True,
                    truncation = False,
                    return_token_type_ids=False,
                    return_attention_mask=True, # We need to attend to padding tokens, so we set this to False
                    return_tensors='pt',
)

example = {'sequence': ['-[ACDEFGHIKLMNPQRSTVWYXXXXXXX]--','--[ADEFHIFGM]---' ]}
encoded = encode(example)
print(encoded)