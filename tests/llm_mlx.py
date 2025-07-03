from mlx_lm import load, generate

# 加载模型和分词器
model, tokenizer = load("/Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

# 设置提示文本
prompt = "What is Python"

# 将提示文本转换为模型所需的格式
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# 生成文本
text = generate(model, tokenizer, prompt=prompt, verbose=True)
print(text)