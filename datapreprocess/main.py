
from fewshot_generator import FewShotSampleGenerator
import torch 
CONFIG = {
    "model_path": "",
    "root_directory": "",
    "blacklist_file": "",
    "output_val_file": "",
    "n_shot": 1,
    "d":"",
    "test_size": 0.2,
    "random_state": 42,
    "batch_size": 2048,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "feature_db_dir": "db1",
    "num_workers": 64,
    "DATASET_CLASSES" :
    {
    "BNCB":["breast tumor","normal breast tissue"],
    "AGGC_2022": [
        "Stroma",
        "normal prostate tissue",
        "The observed differentiation patterns include Gleason 3",
        "The observed differentiation patterns include Gleason 4"
    ],
    "diagset": [
        "normal prostate tissue", "Gleason grade 1", "Gleason grade 3",
        "Gleason grade 4", "Gleason grade 5", "Gleason grade 2"
    ],
    "Gleason": [
        "The observed differentiation patterns include Benign",
        "The observed differentiation patterns include Gleason 3",
        "The observed differentiation patterns include Gleason 5",
        "The observed differentiation patterns include Gleason 4",
        "normal prostate tissue"
    ],
    "VALSET": [
        "Gastric mucosa", "Lamina propria mucosae", "Adventitial tissue",
        "Lamina muscularis mucosae", "Oesophageal tumor", "Areas of ulceration",
        "Submucosal glands", "Regression areas", "Muscularis propria", "Submucosa", "Oesophageal mucosa"
    ],
    "prostate_tu_norm":["benign","prostate tumor"],
    "KICH_sampled":["kidney chromophobe cell carcinoma","normal kidney tissue"],
    "CocaHis":["Metastatic colon cancer in the liver","non-cancer"],
    "CAMEL":["normal","colorectal adenoma"],
    "PAIP19":["Viable-tumor","Non-tumor","Non-viable tumor (intratumoral hemorrhage or necrosis or non-tumor tissue region)"],
    "unipatho":["Hyperplastic Polyp","Normal tissue","Tubular Adenoma, High-Grade dysplasia",
            "Tubular Adenoma, Low-Grade dysplasia","Tubulo-Villous Adenoma, High-Grade dysplasia",
            "Tubulo-Villous Adenoma, Low-Grade dysplasia"],
    "catch":[
        "peripheral nerve sheath tumor",
            "squamous cell carcinoma",
            "trichoblastoma",
            "histiocytoma",
            "bone",
            "cartilage",
            "dermis",
            "epidermis",
            "subcutis",
            "inflammation & necrosis",
            "melanoma",
            "plasmacytoma",
            "mast cell tumor"
    ]
}
}


if __name__ == "__main__":
    generator = FewShotSampleGenerator(CONFIG)
    results = generator.run()  