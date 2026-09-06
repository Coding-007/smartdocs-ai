import os
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from retrieval.rag_chain import get_context, client
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Explicitly wrap LLM and embeddings for RAGAS compatibility
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

def load_test_questions(path : str = 'eval/test_questions.json') -> list:
    with open(path, 'r') as f:
        return json.load(f)

def generate_answer(question: str, context: str) -> str:

    system_prompt = """You are SmartDocs AI.
        Answer ONLY using the context provided below.
        If the answer is not in the context, say:
        'I could not find that in the documents.'
        Always mention the source at the end.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt + f"\n\nContext:\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

def build_eval_dataset(test_questions: list) -> Dataset:

    # Collect the 4 fields RAGAS needs, for every test question
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []

    for item in test_questions:
        question = item["question"]
        print(f"Running: {question}")

        # Run through your REAL RAG pipeline — no shortcuts
        context, sources = get_context(question, history=[])
        answer = generate_answer(question, context)

        questions.append(question)

        # RAGAS expects contexts as a LIST of chunks, not one big string
        contexts_list.append([context])
        answers.append(answer)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths
    })

def run_evaluation():

    test_questions = load_test_questions()
    dataset = build_eval_dataset(test_questions)

    print("\nRunning RAGAS evaluation (this calls GPT multiple times per question)...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return result

if __name__ == "__main__":
    result = run_evaluation()
    print("\n=== EVALUATION RESULTS ===")
    print(result)
