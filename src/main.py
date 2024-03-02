import asyncio
import pandas as pd
from Readfile import read_pdf
from flask_limiter import Limiter
from flask import Flask, request, jsonify
from flask_limiter.util import get_remote_address
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10001 per day", "31 per hour"],
    storage_uri="memory://",
)


def read_csv(file_path="data/Financebench.csv"):
    try:
        df = pd.read_csv(file_path, index_col=False)
        questions = df['question'].to_list()
        doc_links = df['doc_link'].to_list()
        expected_answer = df['answer'].to_list()
        return questions, doc_links, expected_answer
    except KeyError as ke:
        raise ke


def get_document_for_question(doc_links, questions, question):
    return doc_links[questions.index(question)]


def get_text_from_doc(doc_link):
    text = read_pdf(doc_link)
    return text


def train_RAG():
    global tokenizer, model, retriever

    # Initialize the RAG tokenizer and retriever
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    # !!WARNING: below instruction will load datasets from net over 14 hours at 40+MBps speed.
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)

    # Initialize the RAG model for generation
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")


def feed_context(context, question):
    # Tokenizing can be done in two way

    # Tokenize the input context and question
    # input_text = f"Context: {context}\nQuestion: {question}"
    # inputs = tokenizer(input_text, return_tensors="pt")

    # Tokenize the input context and question
    inputs = tokenizer(context, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=512)
    question_ids = tokenizer(question, return_tensors="pt").input_ids

    return inputs, question_ids


def answer(inputs, question_ids):
    # refining the shape if < 5, because n_docs should be multiple of 5.
    if len(inputs['input_ids']) < 5:
        inputs['input_ids'] = inputs['input_ids'].expand(5, -1)
        inputs['attention_mask'] = inputs['attention_mask'].expand(5, -1)

    # Generate the answer
    outputs = model.generate(context_input_ids=inputs['input_ids'],
                             context_attention_mask=inputs['attention_mask'],
                             question_input_ids=question_ids,
                             max_length=50)

    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


async def load_presets(question):
    # this instruction will get the pdf link to which question is asked
    doc_link = get_document_for_question(doc_links, questions, question)

    # this instruction will load the pdf into the python string fro RAG tuning
    context = get_text_from_doc(doc_link)

    # below intruction will tokenize the input as context : question
    inputs, question_ids = feed_context(context, question)
    return inputs, question_ids


async def process_simulate(question):
    inputs, question_ids = await load_presets(question)
    return answer(inputs, question_ids)


@app.route("/answer", methods=["POST"])
@limiter.limit("10/second", override_defaults=False)
async def get_answer():
    content = request.json
    questions = content['question']

    # you can send questions as a list of questions or just one question in string format.
    if type(questions) == list:
        predicted_answer = await asyncio.gather(*[process_simulate(question) for question in questions])
    else:
        predicted_answer = await process_simulate(questions)
    return jsonify({"answer": predicted_answer})

if __name__ == "__main__":
    try:
        print("[*] reading data...\\")
        questions, doc_links, expected_answer = read_csv()
        print("[*] loading data...\\")
        train_RAG()
        print("[*] data onboard complete")
    except KeyError as e:
        raise e
    print("[*] initiate server")
    app.run(host="0.0.0.0", port=8001, debug=True, threaded=True)
