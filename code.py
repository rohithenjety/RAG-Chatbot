import gradio as gr
import openai
import time
import json
import nltk
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the OpenAI API key
openai.api_key = "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"

# Load guidelines from instructions.json file
with open("instructions.json", "r") as file:
    data = json.load(file)
    guidelines = data["guidelines"]

# Tokenizer
# nltk.download('punkt')

# Global variables
rate_limit = 1
last_request_time = 0
messages = [{"role": "system", "content": "You are a company openai, you should be restricted to abide by the guidelines"}]
document_text = ""
document_embeddings = []

def check_guidelines(user_input):
    # Check if the user input matches any of the guidelines
    rohith = word_tokenize(user_input.lower())  # Tokenize and convert to lowercase
    for token in rohith:
        if token.strip().lower() in guidelines:
            return True
    return False

def pdf_to_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_embeddings(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

def find_most_relevant_text(query, document_text, document_embeddings):
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity([query_embedding], document_embeddings)
    most_relevant_index = np.argmax(similarities)
    return document_text.split('\n')[most_relevant_index]

def CustomChatGPT(user_input):
    global last_request_time

    current_time = time.time()
    delay = max(0, rate_limit - (current_time - last_request_time))
    time.sleep(delay)

    last_request_time = current_time
    messages.append({"role": "user", "content": user_input})

    if check_guidelines(user_input):
        # If the user input matches any guideline, print a given statement
        ChatGPT_reply = "You are not supposed to ask this type of questions to me, please ask me any other question, I am here to help you!"
    else:
        relevant_text = find_most_relevant_text(user_input, document_text, document_embeddings)
        messages.append({"role": "user", "content": relevant_text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        ChatGPT_reply = response["choices"][0]["message"]["content"]

    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

def process_pdf(file):
    global document_text, document_embeddings
    document_text = pdf_to_text(file.name)
    document_embeddings = [get_embeddings(page_text) for page_text in document_text.split('\n') if page_text.strip()]
    return "PDF Uploaded and Processed!"

# Create Gradio interface
def gradio_interface(input_text, pdf_file):
    if pdf_file is not None:
        process_pdf(pdf_file)
        return "PDF Uploaded and Processed!"
    else:
        return CustomChatGPT(input_text)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your query here..."), gr.File(label="Upload PDF")],
    outputs="text",
    title="RAG-Powered Chatbot"
)

demo.launch(share=True)
