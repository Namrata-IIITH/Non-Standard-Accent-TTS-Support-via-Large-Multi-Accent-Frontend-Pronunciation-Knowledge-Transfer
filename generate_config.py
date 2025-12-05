import json
import os

# Ensure folder exists
if not os.path.exists("config/edi"):
    os.makedirs("config/edi")

# Base configuration for ONE accent (EDI)
config_data = {
    "experiment_name": "edi_uni_accent",
    "run_description": "Uni-accent Edinburgh baseline",
    "epochs": 6,
    "batch_size": 32,
    "save_step": 5000,
    "batch_group_size": 100,
    "max_seq_len": 600,
    "learning_rate": 0.00005,
    "r": 1,
    "has_postnet": False,
    "has_stopnet": False,
    "has_prenet": False,
    "enc_embedding_dim": 256,
    "dec_embedding_dim": 256,
    "enc_hidden_dim": 512,
    "dec_hidden_dim": 512,
    "post_hidden_dim": 512,

    "src_vocab": "vocab/src.vocab",
    "tgt_vocab": "vocab/tgt.vocab",

    "verbose": True,

    "data": {
        "edi": {
            "corpus_1": {
                "path_src": "dataset/unilex_edi/src-train.txt",
                "path_tgt": "dataset/unilex_edi/tgt-train.txt"
            },
            "valid": {
                "path_src": "dataset/unilex_edi/src-val.txt",
                "path_tgt": "dataset/unilex_edi/tgt-val.txt"
            },
            "test": {
                "path_src": "dataset/unilex_edi/src-test.txt",
                "path_tgt": "dataset/unilex_edi/tgt-test.txt"
            }
        }
    }
}

# Save JSON file
with open("config/edi/edi_uni.json", "w") as f:
    json.dump(config_data, f, indent=4)

print("SUCCESS: config/edi/edi_uni.json created.")
