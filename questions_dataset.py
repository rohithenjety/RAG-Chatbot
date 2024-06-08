
pdf_path = '/mnt/data/ENJETY ROHITH Resume.pdf'
document_text = pdf_to_text(pdf_path)
document_embeddings = [get_embeddings(page_text) for page_text in document_text.split('\n') if page_text.strip()]


dataset = [
    {"question": "What is Rohith's CGPA?", "answer": "9.23"},
    {"question": "Where did Rohith complete his 10th?", "answer": "Bhashyam high school in Tirupati"},
    {"question": "What programming languages does Rohith know?", "answer": "C, Java, python"},
    {"question": "Where is Rohith currently studying?", "answer": "Sree Vidyanikethan Engineering College"},
    {"question": "What certification does Rohith have in cybersecurity?", "answer": "Bug-Bounty in the Tmg.Sec on the OWASP top 10 vulnerabilities"}
]

def evaluate_model(dataset):
    y_true = []
    y_pred = []

    for data in dataset:
        question = data["question"]
        true_answer = data["answer"]

        predicted_answer = CustomChatGPT(question)
        y_true.append(true_answer)
        y_pred.append(predicted_answer)

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    return precision, recall, f1

# Evaluate the model
precision, recall, f1 = evaluate_model(dataset)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
