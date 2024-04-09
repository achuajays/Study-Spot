import torch
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from PIL import Image



device = 'cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)




def predicted(image, max_length=64, num_beams=3):
    # Convert the InMemoryUploadedFile object to a PIL image
    pil_image = Image.open(image)
    pil_image = pil_image.convert('RGB')
    
    # Perform image processing and caption generation
    image = feature_extractor(pil_image, return_tensors="pt").pixel_values.to(device)
    clean_text = lambda x: x.replace('','').split('\n')[0]
    caption_ids = model.generate(image, max_length=max_length)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))
    return caption_text