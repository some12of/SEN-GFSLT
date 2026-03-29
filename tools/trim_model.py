
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

from hftrim.ModelTrimmers import MBartTrimmer

# import utils
import pickle
import gzip
from hftrim.TokenizerTrimmer import TokenizerTrimmer

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
raw_data = load_dataset_file('data/Phonexi-2014T/labels.train')

data = []

for key,value in raw_data.items():
    sentence = value['text']
    # gloss = value['gloss']
    data.append(sentence)
    # data.append(gloss.lower())

# tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="de_DE", tgt_lang="de_DE")
tokenizer = MBartTokenizer.from_pretrained("pretrain_models/MBart_trimmed", src_lang="de_DE", tgt_lang="de_DE")

model = MBartForConditionalGeneration.from_pretrained("pretrain_models/MBart_trimmed")
configuration = model.config

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

# trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

new_tokenizer = tt.trimmed_tokenizer
new_model = mt.trimmed_model

new_tokenizer.save_pretrained('pretrain_models/MBart_trimmed1')
new_model.save_pretrained('pretrain_models/MBart_trimmed1')

## mytran_model
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = new_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = new_model.model.shared

mytran_model.save_pretrained('pretrain_models/mytran1/')











