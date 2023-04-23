from flask import Flask, render_template, send_file, request


import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import *
from torch import autocast
import json

import base64
from io import BytesIO

HUGGINGFACE_TOKEN = "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" 
SENTIMENT_MODEL= AutoModelForSequenceClassification.from_pretrained("DevBeom/dbert_Beomsang", use_auth_token = HUGGINGFACE_TOKEN)
TOKENIZER  = AutoTokenizer.from_pretrained("DevBeom/dbert_Beomsang", use_auth_token = HUGGINGFACE_TOKEN)
DIFFUSOR_MODEL = "DevBeom/stable-diffusion-class4"
STATEMENT = "This image was generated from AI based on your diary"          

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(DIFFUSOR_MODEL, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")


# Start flask app and set to ngrok
app = Flask(__name__, template_folder='.') # period is because default templates folder is /templates

@app.route('/')
def initial():
  return render_template('/static/index.html')

@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  diary = request.form.get('sentiment-input','')
  type_of_art = request.form.get('type-input', '')
  class_labels = ['negative', 'positive']
  SENTIMENT_MODEL.config.id2label = class_labels
  sentiment_model = pipeline('sentiment-analysis', model = SENTIMENT_MODEL, tokenizer = TOKENIZER)
  result = sentiment_model(diary)[0]
  sentiment_weight = ""

  if result['label'] == 'negative' and type_of_art == 'landscape':
    sentiment_weight = ",(NegSenLan:{:.1f}".format(1 + result['score']) + ")"
  elif result['label'] == 'positive' and type_of_art == 'landscape':
    sentiment_weight = ",(I have positive feeling today:{:.1f}".format(1 + result['score']) + ")"
  elif result['label'] == 'negative' and type_of_art == 'portrait':
    sentiment_weight = ",(NegSenPor:{:.1f}".format(1 + result['score']) + ")"
  elif result['label'] == 'positive' and type_of_art == 'portrait':
    sentiment_weight = ",(PosSenPor:{:.1f}".format(1 + result['score']) + ")"
 

  print(sentiment_weight)
  
  print_sentiment = "I have " + result['label'] + " feeling today score: {:.2f}".format(100* + result['score'])

  NEGATIVE_PROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, Typography,watermark, grainy"
  
  prompt = diary + sentiment_weight + "," + type_of_art 
  with autocast("cuda"), torch.inference_mode():
    images = pipe(
            prompt,
            height=512,
            width=512,
            negative_prompt=NEGATIVE_PROMPT,
            num_images_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=None
    ).images 
  
  for img in images:
      buffered = BytesIO()
      img.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue())
      b = "data:image/png;base64," + str(img_str)[2:-1]

  return render_template('/static/index.html', generated_image=b, print_sentiment = print_sentiment, statement=STATEMENT, diary=diary)
app.run(host='0.0.0.0', port=5000)
