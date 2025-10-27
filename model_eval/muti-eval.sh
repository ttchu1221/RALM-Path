#!/bin/bash
set -e  # 出错立即退出

# 设置参数
N=8
base_answer_path="./qwen2"
script_path="/open_model.py"
input_file="" 
model_path="" #your model path 
# 创建目录（如果不存在）
mkdir -p "$base_answer_path"

# 清理旧的分片文件（可选）
rm -f "${base_answer_path}/result_"*.json

echo "Starting $N processes..."

# 并行运行推理任务
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

wait  # 等待所有后台任务完成

# 合并所有结果
merged_file="${base_answer_path}/result.json"
echo "Merging results into $merged_file..."

cat "${base_answer_path}"/result_*.json > "$merged_file"

# 删除中间文件（可选）
echo "Cleaning up temporary files..."
rm -f "${base_answer_path}/result_"*.json

echo "Done."