from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template, request


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_1 = AutoTokenizer.from_pretrained("2003achu/Caption")
model_1 = AutoModelForSeq2SeqLM.from_pretrained("2003achu/Caption")


def generater(prompt):

    batch = tokenizer_1(prompt, return_tensors="pt")
    generated_ids = model_1.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer_1.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]