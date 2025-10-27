import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms
from PIL import Image
import torch
import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import faiss
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

number = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}
number_to_ordinal = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    21: "twenty-first",
    22: "twenty-second",
    23: "twenty-third",
    24: "twenty-fourth",
    25: "twenty-fifth",
    26: "twenty-sixth",
    27: "twenty-seventh",
    28: "twenty-eighth",
    29: "twenty-ninth",
    30: "thirtieth",
    31: "thirty-first"
}

def extract_features_batch(image_paths, feature_extractor, model):

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    # images = Image.open(os.path.join("Pathology",image_paths)).convert('RGB') 
    images = Image.open(image_paths).convert('RGB') 
    inputs = feature_extractor(images=images, return_tensors="pt") 
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs) 
    

    features = outputs.last_hidden_state[:, 0, :]

    features = features.cpu().numpy()

    return features

def build_feature_database(samples, feature_extractor, model,blacklist,batch_size=3072):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    feature_db = defaultdict(dict)
    image_paths = [sample['relative_image']  for sample in samples if sample["relative_image"]not in blacklist]

    captions = [sample['caption'] for sample in samples if sample["relative_image"] not in blacklist]

    for i in range(0, len(image_paths), batch_size):

        batch_paths = image_paths[i:i + batch_size]
        caption = captions[i:i + batch_size]

        def load_image(img_path):
            # full_path = os.path.join("Pathology", img_path)
            full_path = img_path
            return Image.open(full_path).convert('RGB')

        def load_images_parallel(batch_paths, num_workers=8):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(load_image, batch_paths))
            return images
        images = load_images_parallel(batch_paths)
        inputs = feature_extractor(images, return_tensors="pt").to(device)


        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for path, embedding,cap in zip(batch_paths, embeddings,caption):
            feature_db[cap][path] = embedding

    return feature_db

def build_faiss_index(feature_db, target_caption):
    embeddings = []
    img_paths = []

    for unique_id, img_data in feature_db.items():
        if unique_id == target_caption:  
            for img_path, embedding in img_data.items():
                embeddings.append(embedding)
                img_paths.append(img_path)
            break

    embeddings = np.array(embeddings).astype('float32')

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index_flat = faiss.IndexFlatIP(dimension)  
    index_flat.add(embeddings)

    res = faiss.StandardGpuResources()  
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat) 

    return gpu_index, img_paths


def find_topk_faiss(query_embedding, feature_db, target_caption, k=5):

    gpu_index, img_paths = build_faiss_index(feature_db, target_caption)


    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)


    distances, indices = gpu_index.search(query_embedding, k)

    top_k_paths = [img_paths[i] for i in indices[0]]

    return top_k_paths



def select_n_samples_from_same_dataset(model, feature_extractor, feature_db, samples, nums,n_per_batch, blacklist, pred_img):

    query_image_path = pred_img['relative_image']
    query_embedding = extract_features_batch(query_image_path, feature_extractor, model)

    if query_embedding is None:
        raise ValueError(f"Query image path {query_image_path} not found in the feature database.")

    selected_samples = []
    category_samples = defaultdict(list)

    for sample in samples:
        if sample['relative_image'] not in blacklist:
            category_samples[sample['caption']].append(sample)

    def process_category(samples_in_cat):
   
        if len(samples_in_cat) >= nums:

            similar_image_paths = find_topk_faiss(query_embedding, feature_db, samples_in_cat[0]['caption'], k=nums)

            similar_samples = [
                sample for sample in samples 
                if sample['relative_image'] in similar_image_paths and sample['relative_image'] not in blacklist
            ]
            return similar_samples[:nums]
        
        return None  
    for category, samples_in_cat in category_samples.items():
        result = process_category(samples_in_cat)
        if result:
            selected_samples.extend(result)
        
        if len(selected_samples) >= n_per_batch:
            break

    successful_picks = [s for s in selected_samples if s is not None]
    successful_picks.sort(key=lambda x: x['caption'])
    return successful_picks if len(successful_picks) == n_per_batch else None

def generate_human_value(captions,n):
    """Generate a string with image captions for human interaction."""

    examples = [
        f"Example{i}\n<image>\nAnswer:{caption}" for i, caption in enumerate(captions[:-1])
    ]
    example_str = "\n".join(examples)

    ques = f"\nChoose the most likely answer for this image based on what you learned from the examples.\n<image>\n"
    insturct_choice=f"Answer with the option's letter from the given choices directly."

    choices = "\n".join([f"{chr(65 + i)}: {caption}" for i, caption in enumerate(captions[:-1])])

    example_count = len(captions[:-1])
    target_index = len(captions)

    example_ord = number.get(example_count, f"{example_count}")
    target_ord = number_to_ordinal.get(target_index, f"{target_index}th")
    caption_strings = (
            f"You are given {example_ord} example pathological images with their corresponding diagnoses. Based on the visual patterns and features observed in these examples, analyze the {target_ord} image and select the most appropriate diagnosis from the provided options.\n"
            f"{example_str}{ques}{insturct_choice}\nOptions:\n{choices}"
        )
    return caption_strings

def create_combined_sample(samples, sample_id,n):
    if len(samples) < 2:
        print("Not enough samples to create a combined sample.")
        return None
    captions = [sample["caption"] for sample in samples]
    images = [sample["relative_image"] for sample in samples]
    human_value = generate_human_value(captions,n)
    try:
        for i ,caption in enumerate (captions[:-1]):
            if caption == captions[-1]:
                gpt_sample = f"{chr(65 + i)}"
        path_parts = images[0].split('/')
        dataset = path_parts[0]
        
        conversations = [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_sample}
        ]
        combined_sample = {
            "sample_id": sample_id,
            "conversations": conversations,
            "image": images,
            "metadata": {
                "correct":captions[-1],
                "dataset": dataset,
                "question_type": "close"
            },
        }
    except:
        set_trace()

    return combined_sample

def process_file(file_path, root_directory):

    return extract_samples_from_json(file_path, root_directory)

def process_sample(dataset_samples, n,s_blacklist):
    last__samples = []
    caption_samples = defaultdict(list)


    for sample in dataset_samples:
        caption_samples[sample['caption']].append(sample)

    for caption, samples in caption_samples.items():
        seen_relative_images = set()  
        s_samples = []
        for sample in samples:
            relative_image = sample['relative_image']
            if relative_image in s_blacklist and relative_image not in seen_relative_images:
                s_samples.append(sample)
                seen_relative_images.add(relative_image)       
        s_samples.sort(key=lambda x: x['relative_image'])  
        last__samples.extend(s_samples)

    return last__samples

def mutiprocess_samples(samples_dict, n,blacklist,s_blacklist):
    last__samples = []


    with ThreadPoolExecutor() as executor:
        futures = []


        for dataset_samples in samples_dict.values():
            future = executor.submit(process_sample, dataset_samples,n, s_blacklist)
            futures.append(future)


        for future in as_completed(futures):
            partial_last__samples = future.result()
            last__samples.extend(partial_last__samples)

    last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
    for sample in last_samples1:
        blacklist.add(sample['relative_image'])

    return last__samples, blacklist, last_samples1
def extract_samples_from_json(file_path, directory):

    samples = []

    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        return samples

    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)

            for data in data_list:
                caption = data.get('caption')
                image = data.get('image')

                if caption and image:
                    relative_image_path = os.path.relpath(os.path.join(os.path.dirname(file_path), "images", image), start=directory)
                    abs_path = os.path.join(directory, relative_image_path)
                    if os.path.exists(abs_path):
                        samples.append({
                            "relative_image": relative_image_path,
                            "caption": caption,
                            "directory": os.path.dirname(file_path)
                        })
                    else:
                        print(f"Image path does not exist: {relative_image_path}")
                else:
                    print(f"Skipping invalid sample in {file_path}: image={image}, caption={caption}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return samples
def combine_samples_from_same_dataset(samples_dict, n,caption_kinds, sample_counter, blacklist,s_blacklist):
    combined_samples = []

    n_per_batch = n*len(caption_kinds)+1
    last__samples, blacklist, last_samples1 = mutiprocess_samples(samples_dict, n,blacklist,s_blacklist)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    model_name = ""# /path/path-llava
    feature_extractor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device)
    model.eval()

    for dataset_samples in samples_dict.values():
        s = time.time()
        readable_time = datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        print("reading start time:", readable_time)
        feature_db = build_feature_database(dataset_samples, feature_extractor, model,blacklist)
        s1 = time.time()
        readable_time = datetime.fromtimestamp(s1).strftime("%Y-%m-%d %H:%M:%S")
        print("reading end time:", readable_time)
        batch_size = 1000
        while last_samples1:
            batch = random.sample(last_samples1, min(batch_size, len(last_samples1)))
            for last_sample in batch:
                selected_samples = select_n_samples_from_same_dataset(
                    model, feature_extractor, feature_db, dataset_samples, n, n_per_batch - 1, blacklist, pred_img=last_sample
                )
                if not selected_samples or len(selected_samples) < n_per_batch - 1:
                    continue 

                selected_samples.append(last_sample)
                

                combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1, n=n)
                
                if combined_sample:

                    combined_samples.append(combined_sample)
                    sample_counter += 1

            last_samples1 = [x for x in last_samples1 if x not in batch]
                
            
    return combined_samples, sample_counter


def process_all_samples_in_directory(root_directory, blacklist_file, output_val_file,n,random_state=42):

    all_samples_by_dataset = defaultdict(list)
    blacklist = set()  
    s_blacklist = set()
    if not os.path.isfile(blacklist_file):
        print(f"File does not exist: {blacklist_file}")

    print(f"Reading file: {blacklist_file}")
    try:
        with open(blacklist_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            for entry in data_list:
                images = entry.get('image')
                if images and isinstance(images, list): 
                        image = images[-1]
                        if image:  
                            s_blacklist.add(image) 
                else:
                    print(f"Skipping invalid sample in {blacklist_file}: missing or incorrect format of image field")

    except Exception as e:
        print(f"Error reading {blacklist_file}: {e}")

    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        if 'data1.json' in files:
            file_paths.append(os.path.join(subdir, 'data1.json'))

    with Pool(cpu_count()) as pool:
        processed_results = pool.starmap(process_file, [(file_path, root_directory) for file_path in file_paths])


    for result in processed_results:
        all_samples_by_dataset[os.path.dirname(result[0]['directory'])].extend(result)

    sample_counter = 0
    all_combined_samples = []
    

    for dataset_samples in tqdm(all_samples_by_dataset.values()):
        
        caption_kinds = set()

        for item in dataset_samples:
            if 'caption' in item:
                caption_kinds.add(item['caption'])
        combined_samples, sample_counter = combine_samples_from_same_dataset({os.path.dirname(dataset_samples[0]['directory']): dataset_samples}, n,caption_kinds,sample_counter, blacklist,s_blacklist)
        all_combined_samples.extend(combined_samples)

    print("^_^")

    with open(output_val_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)
    print(f"{n}shot dataset saved to {output_val_file}.")
if __name__ == "__main__":
    np.random.seed(42)  
    root_directory = "" #/path/Current directory
    blacklist_file = "/beachmark_close_fewshot.json" #To ensure test sample consistency
    n=1
    output_val_file = os.path.join(root_directory, "bencnmark_fewshot_retireve.json")
    process_all_samples_in_directory(root_directory, blacklist_file, output_val_file,n)

