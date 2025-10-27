#!/bin/bash
set -e  # 


N=8
base_answer_path="./qwen2"
script_path="/open_model.py"
input_file="" 
model_path="" #your model path 

mkdir -p "$base_answer_path"


rm -f "${base_answer_path}/result_"*.json

echo "Starting $N processes..."


for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    answer_path="${base_answer_path}/result_${chunk_id}.json"
    
    echo "Launching chunk $chunk_id on GPU $chunk_id..."
    CUDA_VISIBLE_DEVICES="$chunk_id" python3 "$script_path" \
        --answers-file "$answer_path" \
        --num-chunks "$N" \
        --chunk-idx "$chunk_id" \
        --input_file "$input_file"\
        --model_path "$model_path" &
done

wait 

merged_file="${base_answer_path}/result.json"
echo "Merging results into $merged_file..."

cat "${base_answer_path}"/result_*.json > "$merged_file"


echo "Cleaning up temporary files..."
rm -f "${base_answer_path}/result_"*.json

echo "Done."