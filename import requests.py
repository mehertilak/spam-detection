import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

img_url = 'https://nataliabrochini.com/wp-content/uploads/2022/05/dog-bench-portrait-professional-photography-natalia-brochini.jpg'
raw_image = Image.open('/content/road-car-vehicle-170811.jpg').convert('RGB')

question = "which  the car is?"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

