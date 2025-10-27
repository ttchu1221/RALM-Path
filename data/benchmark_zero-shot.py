import os
import json
import logging
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from pdb import set_trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qc1 = ["<image>\nDescribe the image concisely.",
    "<image>\nWhat is displayed in the histology image?"
]

DATASET_CLASSES = {
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

def extract_samples_from_json(file_path, directory):
    samples = []

    if not os.path.isfile(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return samples

    logger.info(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        for data in data_list:
            caption = data.get('caption')
            image = data.get('image')

            if caption and image:
                relative_image_path = os.path.relpath(
                    os.path.join(os.path.dirname(file_path), "images", image),
                    start=directory
                )
                abs_path = os.path.join(directory, relative_image_path)
                if os.path.exists(abs_path):
                    samples.append({
                        "relative_image": relative_image_path,
                        "caption": caption
                    })
                else:
                    logger.warning(f"Image path does not exist: {relative_image_path}")
            else:
                logger.warning(f"Skipping invalid sample in {file_path}: image={image}, caption={caption}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return samples


def generate_multiple_choice_data(caption, options, question_template):
    augmentations = []

    choices_str = "\n".join([f"{chr(65 + i)}. {cap}" for i, cap in enumerate(options)])
    question = f"{question_template}\nChoices:\n{choices_str}\nAnswer with the option's letter from the given choices directly."
    for i ,option in enumerate(options):
        if option == caption:
            answer_letter = chr(65 + i)
            augmentations.append({
                "human_value": question,
                "gpt_response": answer_letter,
                "metadata": {
                    "real_answer": caption,
                    "dataset": None,  
                    "question_type": "close"
                }
            })
    return augmentations





def create_combined_samples(sample, sample_id_base):
    caption = sample["caption"]
    relative_image_path = sample["relative_image"]
    img_dataset = relative_image_path.split("/")[0]
  
    question_index =1
    question_template = qc1[question_index]
    samples = []
    if question_index ==1:

        options = DATASET_CLASSES.get(img_dataset)

        if not options or caption not in options:
 
            logger.warning(f"No valid options found for dataset: {img_dataset}")
            return []
        augmentations = generate_multiple_choice_data(caption, options, question_template)
        for i, aug in enumerate(augmentations):
            conversations = [
                {"from": "human", "value": aug["human_value"]},
                {"from": "gpt", "value": aug["gpt_response"]}
            ]
            samples.append({
                "sample_id": f"{sample_id_base}",
                "conversations": conversations,
                "image": [relative_image_path],
                "metadata": {
                    "real_answer": aug["metadata"]["real_answer"],
                    "dataset": img_dataset,
                    "question_type": "close"
                }
            })
            sample_id_base+=1
        return samples,sample_id_base
    else:
        
        conversations = [
            {"from": "human", "value": question_template},
            {"from": "gpt", "value": caption}
        ]
        samples.append({
            "sample_id": f"{sample_id_base}",
            "conversations": conversations,
            "image": [relative_image_path],
            "metadata": {
                "dataset": img_dataset,
                "question_type": "open"
            }
        })
        sample_id_base+=1
        return samples,sample_id_base


def process_file(file_path, root_directory):

    return extract_samples_from_json(file_path, root_directory)


def process_all_samples_in_directory(root_directory, output_train_file, samples_per_dataset=5000):
    all_samples_by_dataset = defaultdict(list)

    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        if 'data.json' in files:
            file_paths.append(os.path.join(subdir, 'data.json'))

    with Pool(cpu_count()) as pool:
        processed_results = pool.starmap(process_file, [(file_path, root_directory) for file_path in file_paths])

    for result in processed_results:
        dataset_name = result[0]["relative_image"].split("/")[0]
        all_samples_by_dataset[dataset_name].extend(result)
    sampled_datasets = []
    for dataset_name, samples in all_samples_by_dataset.items():
        if len(samples) > samples_per_dataset:
            sampled_datasets.extend(random.sample(samples, samples_per_dataset)) 
        else:
            sampled_datasets.extend(samples)

    sample_counter = 1
    all_combined_samples = []
    for sample in sampled_datasets:
        augmented_samples, sample_id_base = create_combined_samples(sample, sample_counter)
        all_combined_samples.extend(augmented_samples)
        sample_counter = sample_id_base

    if not all_combined_samples:
        logger.warning("No samples were processed because all datasets had insufficient samples.")

    with open(output_train_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)
    logger.info(f"zero-shot-dataset saved to {output_train_file}.")

if __name__ == "__main__":
    import random
    random.seed(1013)
    root_directory = "" #/path/your_datasetpath
    output_train_file = os.path.join(root_directory, "beachmark_close.json")

    process_all_samples_in_directory(
        root_directory=root_directory,
        output_train_file=output_train_file,
        samples_per_dataset=50
    )

