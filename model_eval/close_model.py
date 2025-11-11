
import base64
from pdb import set_trace
import os 
from openai import OpenAI
import time
from PIL import Image

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


    with Image.open(image_path) as img:

        import io
        buffered = io.BytesIO()
        img.save(buffered, format=img.format) 

     
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

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
            user_content.append({"type": "image_url", "image_url":{"url": f"data:image/{image_type};base64,{url}"}})
            user_content.append({"type": "text", "text": f"Answer:{parts[1]}"})
        user_content.append({"type": "text", "text": f"Choose the most likely answer for this image based on what you learned from the examples."})    
        
        test_case_question =examples_and_test_case[1].strip()
        
        user_content.append({"type": "image_url","image_url":{"url":f"data:image/{image_type};base64,{image_urls[-1]}"}})
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
def model_respoce(key,datas):
    client = OpenAI(
        base_url='',

        api_key=key
    )
    responses = []
    img_path = []
    text_list = []
    for data in datas:
        image_path = data["image"]
        text = data["conversations"][0]["value"].replace("\n<image>","")
        text_list.append(text)
        img_path.append(image_path)
    
        img_urls, it = image_format(image_path)
        message=construct_message(text, img_urls, it)
        time.sleep(1)
        try:
            response = client.chat.completions.create(
                model="",
                temperature = 0,
                messages=message,
                max_tokens=100,
            )
        except Exception as e:
               print(e)
        responses.append(response.choices[0].message.content)
    return responses,img_path,text_list
import json
input_file = "" # test your json 
key =""    # sk-xxx replace your key  
with open(input_file, "r", encoding="utf-8") as f:
    datas = json.load(f)

print("start eval")
response,image_path,text_list = model_respoce(key,datas)

def conv(gpt_response, human_text, img_path):
    return [[
        {"from": "human", "value": human_text},
        {"from": "gpt", "value": gpt_response},
        {"image": img_path} 
    ]]

data = []

for gpt_response, img_path, human_text in zip(response, image_path, text_list):
    a = conv(gpt_response, human_text, img_path)
    data.append(a)

output_file = "./result.json" 
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print("end eval")