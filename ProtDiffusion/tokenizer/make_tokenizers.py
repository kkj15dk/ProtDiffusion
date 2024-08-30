from dataclasses import dataclass

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast


@dataclass
class TrainingConfig:
    pad_to_multiple_of = 16  # the model needs padding to be a multiple of 8, for the U-Net architecture in use
    padding_side = "right"  # where to put the padding tokens in padding to multiple of.
    bos_token = "[" # beginning of sequence token
    eos_token = "]" # end of sequence token
    unk_token = "X"  # unknown token
    pad_token = "-"
    output_dir = "protein_tokenizer_new"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    repo_id = "kkj15dk/protein_tokenizer_new"  # the name of the repository to create on the HF Hub

config = TrainingConfig()

# tokenizer for the decoder
tokenizer = Tokenizer(WordLevel(unk_token=config.unk_token))
tokenizer.pre_tokenizer = Split("", behavior='removed')
tokenizer.post_processor = TemplateProcessing(
    single=config.bos_token + " $A " + config.eos_token,
    special_tokens=[(config.pad_token, 0),
                    (config.bos_token, 1), 
                    (config.eos_token, 2),
                    ])

tokenizer.decoder = decoders.Fuse()

trainer = WordLevelTrainer(special_tokens=[config.pad_token,
                                           config.bos_token,
                                           config.eos_token,
                                           config.unk_token,
                                           ])

files = ['characters.txt']
tokenizer.train(files, trainer)
tokenizer.save("tokenizer.json")


# tokenizer for the encoder
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json",
                                         bos_token=config.bos_token,
                                         eos_token=config.eos_token,
                                         unk_token=config.unk_token,
                                         pad_token=config.pad_token,
                                         padding_side=config.padding_side,
                                         clean_up_tokenization_spaces=False,
)

def encode(example):
    return fast_tokenizer(example['text'],
    # return tokenizer(example['sequence'],
                    padding = True,
                    pad_to_multiple_of = config.pad_to_multiple_of,
                    return_token_type_ids=False,
                    return_attention_mask=True, # We need to attend to padding tokens, so we set this to False
)

example = {'text': 'ACDEFGHIKLMNOPQRSTUVWY'}
encoded = encode(example)
print(encoded)

fast_tokenizer.push_to_hub(config.repo_id)