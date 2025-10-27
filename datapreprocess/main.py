import argparse
import torch
from fewshot_generator import FewShotSampleGenerator

def get_config_from_args():
    parser = argparse.ArgumentParser(description="Few-shot sample generator configuration")

    # 常用参数
    parser.add_argument("--model_path", type=str, default="", help="Path to the pretrained model")
    parser.add_argument("--root_directory", type=str, default="", help="Root directory of the dataset")
    parser.add_argument("--blacklist_file", type=str, default="", help="Path to blacklist file")
    parser.add_argument("--output_val_file", type=str, default="", help="Output file for validation results")
    parser.add_argument("--n_shot", type=int, default=1, help="Number of shots per class")
    parser.add_argument("--d", type=str, default="", help="select similarity (L2 or cosine)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--feature_db_dir", type=str, default="db", help="Feature database directory")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers for dataloaders")
    parser.add_argument("--dataset", type=str, default="BNCB", help="Dataset name, e.g. BNCB, KICH_sampled, etc.")

    args = parser.parse_args()

    CONFIG = {
        "model_path": args.model_path, 
        "root_directory": args.root_directory,
        "blacklist_file": args.blacklist_file,
        "output_val_file": args.output_val_file,
        "n_shot": args.n_shot,
        "d": args.d,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "batch_size": args.batch_size,
        "device": args.device,
        "feature_db_dir": args.feature_db_dir,
        "num_workers": args.num_workers,
        "DATASET_CLASSES": {
            "BNCB": ["breast tumor", "normal breast tissue"],
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
            "prostate_tu_norm": ["benign", "prostate tumor"],
            "KICH_sampled": ["kidney chromophobe cell carcinoma", "normal kidney tissue"],
            "CocaHis": ["Metastatic colon cancer in the liver", "non-cancer"],
            "CAMEL": ["normal", "colorectal adenoma"],
            "PAIP19": [
                "Viable-tumor", "Non-tumor", 
                "Non-viable tumor (intratumoral hemorrhage or necrosis or non-tumor tissue region)"
            ],
            "unipatho": [
                "Hyperplastic Polyp", "Normal tissue", "Tubular Adenoma, High-Grade dysplasia",
                "Tubular Adenoma, Low-Grade dysplasia", "Tubulo-Villous Adenoma, High-Grade dysplasia",
                "Tubulo-Villous Adenoma, Low-Grade dysplasia"
            ],
            "catch": [
                "peripheral nerve sheath tumor", "squamous cell carcinoma", "trichoblastoma", "histiocytoma",
                "bone", "cartilage", "dermis", "epidermis", "subcutis",
                "inflammation & necrosis", "melanoma", "plasmacytoma", "mast cell tumor"
            ]
        }
    }

    return CONFIG

 
if __name__ == "__main__":
    CONFIG = get_config_from_args()
    generator = FewShotSampleGenerator(CONFIG)
    results = generator.run()