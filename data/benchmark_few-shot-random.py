import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from multiprocessing import Pool, cpu_count
from pdb import set_trace
import time

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

def select_n_samples_from_same_dataset(samples,nums,n_per_batch,caption_kinds, blacklist):
    category_samples = defaultdict(list)
    for sample in samples:
        if sample['relative_image'] not in blacklist:
            category_samples[sample['caption']].append(sample)
    selected_samples = []
    
    for category, samples_in_cat in category_samples.items():
        caption_groups = defaultdict(list)
        last_samples =[]
        for sample in samples_in_cat:
            caption_groups[sample['caption']].append(sample)

        found_pair = False
        for caption, group in caption_groups.items():
            if len(group) >= nums:
                selected_samples.extend(random.sample(group, nums))  
                found_pair = True
                break
        
        if not found_pair:
            selected_samples.append(None)  
        if len(selected_samples) == n_per_batch:
            break
    successful_picks = [s for s in selected_samples if s is not None]
    return successful_picks if len(successful_picks) == n_per_batch  else None
def generate_human_value(captions,n):

    examples = [
        f"Example{i}\n<image>\nAnswer:{caption}" for i, caption in enumerate(captions[:-1])
    ]
    example_str = "\n".join(examples)

    ques = f"\nChoose the most likely answer for this image based on what you learned from the examples.\n<image>\n"
    insturct_choice=f"Answer with the option's letter from the given choices directly."

    choices = "\n".join([f"{chr(65 + i)}: {caption}" for i, caption in enumerate(captions[:-1:n])])

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
    for i ,caption in enumerate (captions[:-1:n]):
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

    return combined_sample
def process_file(file_path, root_directory):
    return extract_samples_from_json(file_path, root_directory)

def combine_samples_from_same_dataset(samples_dict, n,caption_kinds, sample_counter, blacklist,s_blacklist):
    combined_samples = []

    n_per_batch = n*len(caption_kinds)+1
    last__samples = []
    for dataset_samples in samples_dict.values():
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
            for last_samples in last__samples:
                blacklist.add(last_samples['relative_image'])
            last_samples1 = [dict(s) for s in set(frozenset(d.items()) for d in last__samples)]
    for dataset_samples in samples_dict.values():
        c = 0
        s = time.time()
        
        while True:
            c += 1
            selected_samples = select_n_samples_from_same_dataset(dataset_samples, n,n_per_batch-1,caption_kinds, blacklist)
            if not selected_samples or  not last_samples1 or len(selected_samples) < n_per_batch - 1:
                break
            last_sample = random.choice(last_samples1)

            selected_samples.append(last_sample)
            last_samples1.remove(last_sample)
            combined_sample = create_combined_sample(selected_samples, sample_id=sample_counter + 1,n=n)
            if combined_sample:
                combined_samples.append(combined_sample)
                sample_counter += 1
            
            if c % 1000 == 0:
                e = time.time()
                print(e - s)
   
    return combined_samples, sample_counter
def process_all_samples_in_directory(root_directory, blacklist_file,output_val_file,n,random_state=42):
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
    # Collect all file paths first
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
    

    for dataset_samples in all_samples_by_dataset.values():
        
        caption_kinds = set()

        for item in dataset_samples:
            if 'caption' in item:
                caption_kinds.add(item['caption'])
        combined_samples, sample_counter = combine_samples_from_same_dataset({os.path.dirname(dataset_samples[0]['directory']): dataset_samples}, n,caption_kinds,sample_counter, blacklist,s_blacklist)
        all_combined_samples.extend(combined_samples)

    print("^_^")

    with open(output_val_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)

    print(f"valing_few_shot_{n} set saved to {output_val_file}.")
if __name__ == "__main__":

    import numpy as np
    np.random.seed(42) 
    root_directory = "" #/path/Current directory
    blacklist_file = "/beachmark_close.json"
    n=1 
    output_val_file = os.path.join(root_directory, "beachmark_close_1shot.json")
    process_all_samples_in_directory(root_directory, blacklist_file,output_val_file,n)