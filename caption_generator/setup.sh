# environment
conda create --name caption_generator python=3.10 -y
conda activate caption_generator
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade transformers

# model download
export HF_ENDPOINT=https://hf-mirror.com
cd checkpoints
pip install en_core_web_sm-2.3.0.tar.gz
# llama 3
cd llama3-8b
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --token hf_xxxxxxxxxxxxxxxxxxxxxxxxx --local-dir . --local-dir-use-symlinks False
# Mistral
cd ../mistral-7b
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --token hf_xxxxxxxxxxxxxxxxxxxxxxxxx --local-dir . --local-dir-use-symlinks False
