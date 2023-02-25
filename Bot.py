from transformers import GPT2LMHeadModel, GPT2Tokenizer

# تهيئة النموذج والتوكنايزر
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# تعريف النص المحفز (prompt) وتحويله إلى input_ids
prompt = "Hello, I am a Chatbot"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# توليد الإجابة
output = model.generate(input_ids=input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# تحويل output إلى نص
response = tokenizer.decode(output[0], skip_special_tokens=True)

# طباعة النص الإجابي
print(response)
