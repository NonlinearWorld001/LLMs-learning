from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\tokenizer\trained_tokenizer")

messages = [
    {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
    {"role": "user", "content": '是椭圆形的'},
    {"role": "assistant", "content": '456'},
    {"role": "user", "content": '456'},
    {"role": "assistant", "content": '789'}]
result = tokenizer.apply_chat_template(messages, tokenize=True)
print(result)