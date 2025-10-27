import os
import json
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import faiss

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    ViTModel,
    ViTFeatureExtractor
)
from sklearn.metrics.pairwise import cosine_similarity


class FewShotSampleGenerator:


    def __init__(self, config: Dict):
        self.config = config
        self.feature_extractor = None
        self.model = None
        self.samples_by_dataset = {}
        self.blacklist = set()

        self.feature_dbs = {}  
        

        random.seed(config["random_state"])
        np.random.seed(config["random_state"])
        torch.manual_seed(config["random_state"])


        os.makedirs(os.path.dirname(config["output_val_file"]), exist_ok=True)
        os.makedirs(config["feature_db_dir"], exist_ok=True)

    def load_data(self):

        print("Loading samples from root directory...")
        self.samples_by_dataset = self._collect_all_samples(self.config["root_directory"])
        if self.config["blacklist_file"]:
            self.blacklist = self._load_blacklist(self.config["blacklist_file"])
        print(f"Found {len(self.samples_by_dataset)} datasets.")

    def _collect_all_samples(self, root_dir: str) -> Dict[str, List[Dict]]:

        file_paths = []
        for root, _, files in os.walk(root_dir):
            if "data1.json" in files:
                file_paths.append(os.path.join(root, "data1.json"))

        with Pool(cpu_count()) as pool:
            results = pool.starmap(self._load_json_file, [(fp, root_dir) for fp in file_paths])

        samples_by_dataset = defaultdict(list)
        for result in results:
            if result:
                dataset_key = result[0]["relative_image"].split('/')[0]
                samples_by_dataset[dataset_key].extend(result)
        return dict(samples_by_dataset)

    def _load_json_file(self, file_path: str, base_dir: str) -> List[Dict]:
     
        samples = []
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            return samples

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            for item in data_list:
                caption = item.get("caption")
                image_name = item.get("image")
                if not caption or not image_name:
                    continue
                image_path = os.path.join(os.path.dirname(file_path), "images", image_name)
                rel_path = os.path.relpath(image_path, start=base_dir)
                abs_path = os.path.join(base_dir, rel_path)
                if not os.path.exists(abs_path):
                    print(f"[WARN] Image not found: {abs_path}")
                    continue
                samples.append({
                    "relative_image": rel_path,
                    "caption": caption,
                    "directory": os.path.dirname(file_path)
                })
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
        return samples

    def _load_blacklist(self, blacklist_path: str) -> set:
  
        blacklist = set()
        if not os.path.exists(blacklist_path):
            print(f"[WARN] Blacklist file not found: {blacklist_path}")
            return blacklist

        try:
            with open(blacklist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data:
                images = entry.get("image", [])
                if isinstance(images, list) and images:
                    blacklist.add(images[-1])
        except Exception as e:
            print(f"[ERROR] Failed to load blacklist: {e}")
        return blacklist

    def load_model(self):

        print("Loading vision model...")
        model_path = self.config["model_path"]
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_path)
        self.model = CLIPVisionModel.from_pretrained(model_path).to(self.config["device"])
        self.model.eval()

    def _extract_features_batch(self, image_paths: List[str]) -> List[np.ndarray]:

        def load_image(path):
            full_path = os.path.join(self.config["root_directory"], path)
            return Image.open(full_path).convert("RGB")

        images = [load_image(p) for p in image_paths]
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.config["device"])

        with torch.no_grad():
            outputs = self.model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        return features

    def _build_faiss_index(self, embeddings: List[np.ndarray]) -> faiss.Index:

        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def _find_knn_faiss(
        self,
        query_embedding: np.ndarray,
        feature_db: Dict[str, Dict[str, np.ndarray]],
        target_caption: str,
        k: int = 5
    ) -> List[str]:

        candidates = []
        for img_path, embedding in feature_db.get(target_caption, {}).items():
            candidates.append((img_path, embedding))

        if len(candidates) < k:
            return [p for p, _ in candidates]

        embeddings = [emb for _, emb in candidates]
        index = self._build_faiss_index(embeddings)

        query_vec = np.array(query_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)

        _, indices = index.search(query_vec, k)
        return [candidates[i][0] for i in indices[0]]
    def _process_sample(self,dataset_samples):
        last__samples = []
        caption_samples = defaultdict(list)
        
        for sample in dataset_samples:
            caption_samples[sample['caption']].append(sample)

        for caption, samples in caption_samples.items():

            random.shuffle(samples)
            cutoff_index = max(1, int(len(samples) * 0.98))

            last__samples.extend(samples[cutoff_index:])
        return last__samples
    def _build_feature_database(self, dataset_name: str, samples: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
        if self.config["n_shot"]== 0:
            init_samples = self._process_sample(samples)
            for sample in init_samples:
                self.blacklist.add(sample["relative_image"])
            
  
        if self.config["d"] == "L2":
            cache_path = os.path.join(self.config["feature_db_dir"], f"feature_db_{dataset_name}_L2.pkl")
        else:
            cache_path = os.path.join(self.config["feature_db_dir"], f"feature_db_{dataset_name}_cos.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached feature DB for {dataset_name}...")
            return joblib.load(cache_path)

        print(f"Building feature DB for {dataset_name}...")
        feature_db = defaultdict(dict)
   
        valid_samples = [s for s in samples if s["relative_image"] not in self.blacklist]

        image_paths = [s["relative_image"] for s in valid_samples]
        captions = [s["caption"] for s in valid_samples]

        batch_size = self.config["batch_size"]
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_captions = captions[i:i + batch_size]
            features = self._extract_features_batch(batch_paths)

            for path, feat, cap in zip(batch_paths, features, batch_captions):
                feature_db[cap][path] = feat

        joblib.dump(feature_db, cache_path)
        print(f"Saved feature DB to {cache_path}")
        return dict(feature_db)

    def _select_similar_samples(
        self,
        feature_db: Dict[str, Dict[str, np.ndarray]],
        dataset_samples: List[Dict],
        n: int,
        pred_img: Dict
    ) -> Optional[List[Dict]]:

        query_path = pred_img["relative_image"]
        query_embedding = self._extract_features_batch([query_path])[0]
        s_path = []
        num = 0 
        for caption in feature_db.keys():
            num +=1
            initial_results = self._find_knn_faiss(query_embedding, feature_db, caption, k=n)
            s_path.extend(initial_results)


        selected = []

        for path in s_path:
            if path in self.blacklist:
                continue
            matched = next((s for s in dataset_samples if s["relative_image"] == path), None)
            if matched:
                selected.append(matched)
            if len(selected) >= num*n:
                break
        return selected if len(selected) == num*n else None

    def _generate_prompt(self, captions: List[str], dataset: str) -> str:
        example_count = len(captions) - 1
        target_index = len(captions)

        example_word = {i: word for i, word in enumerate([
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty"
        ], 1)}.get(example_count, str(example_count))

        number_to_ordinal = {i: f"{i}th" for i in range(1, 32)}
        for i, suffix in [(1, "st"), (2, "nd"), (3, "rd")]:
            number_to_ordinal[i] = f"{i}{suffix}"
        number_to_ordinal[21] = "twenty-first"
        number_to_ordinal[22] = "twenty-second"
        number_to_ordinal[23] = "twenty-third"
        number_to_ordinal[31] = "thirty-first"
        target_ord = number_to_ordinal.get(target_index, f"{target_index}th")

        examples = [
            f"Example{i+1}\n<image>\nAnswer: {captions[i]}"
            for i in range(len(captions) - 1)
        ]
        example_str = "\n".join(examples)

        choices = "\n".join([
            f"{chr(65 + i)}: {cls}" for i, cls in enumerate(self.config["DATASET_CLASSES"][dataset])
        ])
        if self.config["n_shot"] !=0:
            instruction = (
                f"You are given {example_word} example pathological images with their corresponding diagnoses. "
                f"Based on the visual patterns and features observed in these examples, analyze the {target_ord} image "
                f"and select the most appropriate diagnosis from the provided options.\n"
                f"{example_str}\n\n"
                f"Choose the most likely answer for this image based on what you learned from the examples.\n<image>\n"
                f"Answer with the option's letter from the given choices directly.\n"
                f"Options:\n{choices}"
            )
        else:
            instruction = (f"<image>\nWhat is displayed in the histology image?\n"
                           f"Answer with the option's letter from the given choices directly.\n"
                           f"Choices:\n{choices}")
        return instruction

    def _create_combined_sample(self, samples: List[Dict], sample_id: int) -> Optional[Dict]:
        if len(samples) < 2 and self.config["n_shot"]!=0:
            return None
        captions = [s["caption"] for s in samples]
        images = [s["relative_image"] for s in samples]
        dataset_name = images[0].split("/")[0]
        
        prompt = self._generate_prompt(captions, dataset_name)

        try:
            answer_idx = self.config["DATASET_CLASSES"][dataset_name].index(captions[-1])
            correct_answer = chr(65 + answer_idx)
        except ValueError:
            print(f"[WARN] Caption not in class list: {captions[-1]}")
            return None

        return {
            "sample_id": sample_id,
            "image": images,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": correct_answer}
            ],
            "metadata": {
                "correct": captions[-1],
                "dataset": dataset_name,
                "question_type": "close"
            }
        }

    def process_dataset(self, dataset_name: str, dataset_samples: List[Dict]) -> List[Dict]:
 
        print(f"Processing dataset: {dataset_name}")
        
        feature_db = self._build_feature_database(dataset_name, dataset_samples)
        # cutoff = max(1, int(len(dataset_samples) * 0.8))
        # candidate_samples = dataset_samples[cutoff:]
        
        if self.config["n_shot"] < 100:
            seen = set()
            candidate_samples = []

            for s in dataset_samples:
                rel_img = s["relative_image"]
                # 同时满足：在黑名单中，且之前未见过
                if rel_img in self.blacklist and rel_img not in seen:
                    candidate_samples.append(s)
                    seen.add(rel_img)
        # if dataset_name == "PAIP19":
        #     set_trace()
        combined_samples = []
        sample_counter = 0
       
        for pred_sample in candidate_samples:
            similar_samples = []
            if self.config["n_shot"] != 0 :
                similar_samples = self._select_similar_samples(
                    feature_db, dataset_samples,self.config["n_shot"], pred_sample
                )
                #set_trace()
                if not similar_samples:
                    continue
            similar_samples.append(pred_sample)
            #set_trace()
            combined = self._create_combined_sample(similar_samples, sample_counter + 1)
            if combined:
                combined_samples.append(combined)
                sample_counter += 1

        return combined_samples

    def run(self) -> List[Dict]:
 
        print("🚀 Starting Few-shot Sample Generation Pipeline...")
        print(f"🕒 starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.load_data()
        self.load_model()

        all_combined_samples = []
        for dataset_name, dataset_samples in self.samples_by_dataset.items():
                samples = self.process_dataset(dataset_name, dataset_samples)
                all_combined_samples.extend(samples)


        output_path = self.config["output_val_file"]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_combined_samples, f, indent=4, ensure_ascii=False)

        print(f"✅ Generated {len(all_combined_samples)} few-shot samples.")
        print(f"Saved to {output_path}")
        print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return all_combined_samples