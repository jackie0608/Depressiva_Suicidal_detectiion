from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained('models/t5/t5').to('cuda')

if __name__ == "__main__":
	print("Please wait while the analyzer is being initialized.")
	text = input("Input text to analyze sentiment: ")
	while text:
		torch.cuda.empty_cache()

		inputs = tokenizer.encode_plus("sst2 sentence: "+ text, padding='max_length', max_length=512, return_tensors='pt').to('cuda')
		outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
		prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

		if prediction == 'positive' or prediction == 0 or prediction == '0':
			sentiments = "Non-Suicide"
		else:
			sentiments = "Suicide"


		print(sentiments)
		text = input("Input sentiment to analyze: ")


