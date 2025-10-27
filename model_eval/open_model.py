#!/usr/bin/env python3
import base64
import os
import json
import argparse
from tqdm import tqdm
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("--answers-file", type=str, required=True)
parser.add_argument("--num-chunks", type=int, default=8)
parser.add_argument("--chunk-idx", type=int, required=True)
parser.add_argument("--input_file", type=str,default="")
parser.add_argument("--model_path", type=str,default="")
args = parser.parse_args()

input_file = args.input_file
model_path = args.model_path
# input_file = "./bencnmark_oneshot_retireve_1.json"
# model_path = "./qwen2vl"

def image_format(image_path):
    img_list = []
    for img in image_path:
        base64_image= image_to_base64(img)
        img_list.append(base64_image)
    if img_list[0].startswith("iVBORw0KG"):
        it = "png"
    else:
        it = "jpeg"
    return img_list,it
def image_to_base64(image_path):

    img =image_path
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string



def construct_message(text, image_urls, image_type):
    if len(image_urls) >1:
        user_content = []
        examples_and_test_case = text.split("Choose the most likely answer for this image based on what you learned from the examples.")

        message = []
        example_texts = examples_and_test_case[0].split("\nExample")

        for i, (text,url) in enumerate(zip(example_texts[1:], image_urls[:-1])):
            parts = text.split("\nAnswer:")
            user_content.append({"type": "text", "text": f"Example{i}:"})
            user_content.append({"type": "image", "image": f"data:image/{image_type};base64,{url}"
                })
            user_content.append({"type": "text", "text": f"Answer:{parts[1]}"})
        user_content.append({"type": "text", "text": f"Choose the most likely answer for this image based on what you learned from the examples."})    
        
        test_case_question =examples_and_test_case[1].strip()
        
        user_content.append({"type": "image","image": f"data:image/{image_type};base64,{image_urls[-1]}"
                })
        user_content.append({"type": "text", "text": test_case_question})

        message.append({"role": "user", "content": user_content})
    else:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
        for base64_image in image_urls:
            message[0]['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{base64_image}"
                }
            })

    return message

def conv(gpt_response, human_text, img_path):
    return [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": gpt_response},
            {"image": img_path}
        ]
    
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def ans(datas):
    results = []
    for data in tqdm(datas):
        image_path = data["image"]
        text = data["conversations"][0]["value"].replace("<image>\n", "")
        img_urls, it = image_format(image_path)
        messages = construct_message(text, img_urls, it)

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=10,do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        results.append(conv(output_text[0],text,image_path))
    return results


if __name__ == "__main__":

    with open(input_file, "r", encoding="utf-8") as f:
        all_datas = json.load(f)


    total = len(all_datas)
    per_chunk = (total + args.num_chunks - 1) // args.num_chunks
    start = args.chunk_idx * per_chunk
    end = min(start + per_chunk, total)
    sub_datas = all_datas[start:end]
    results = ans(sub_datas)

    with open(args.answers_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")