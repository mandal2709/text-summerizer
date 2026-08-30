from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates #UI
from fastapi.staticfiles import StaticFiles #UI

# Initialize FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="text summarization using T5.",
    version="1.0",
)

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

#templating
templates = Jinja2Templates(directory=".")

# Input schema for dialogue => String
class DialogueInput(BaseModel):
    dialogue: str   

import re

#clean data function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) #lines
    text = re.sub(r"\s+", " ", text) #spaces
    text = re.sub(r"<.*?>", " ", text) #html tags
    text = text.strip().lower()
    return text

# summerization function
def summerize_dialogue(dialogue):
    dialogue = clean_data(dialogue) #clean

    #tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    #generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    #token ids convert to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# Define the API endpoint for summarization
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summerize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )