import argparse
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from download_model import download_model
import csv

MODELS = {
    "mT5": "mT5",
    "BART": "mBART50",
    "Llama": "Llama-3.2-1B-Instruct"
}
LANGUAGE_CODES = {
    "English": "en_XX",
    "French": "fr_XX"
}
EXPERIMENTS = {
    "mT5": ["zero-shot", "few-shot fine-tuning", "full fine-tuning"],
    "mBART50": ["zero-shot", "1-shot", "few-shot fine-tuning", "full fine-tuning"],
    "Llama-3.2-1B-Instruct": ["zero-shot", "2-shot", "few-shot fine-tuning", "full fine-tuning"]
}

def get_max_lengths(model_name):
    """Return max_input and max_output based on model type."""
    if model_name == "mT5":
        return 512, 128
    elif model_name in ["mBART50", "Llama-3.2-1B-Instruct"]:
        return 1024, 128
    else:
        raise ValueError(f"Unknown model: {model_name}")

def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_1, rouge_2, rouge_L = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        rouge_1.append(scores["rouge1"].fmeasure)
        rouge_2.append(scores["rouge2"].fmeasure)
        rouge_L.append(scores["rougeL"].fmeasure)

    return {
        "ROUGE-1": np.mean(rouge_1),
        "ROUGE-2": np.mean(rouge_2),
        "ROUGE-L": np.mean(rouge_L),
    }

def summarize_text_mt5(text, model, tokenizer, max_input, max_output):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=max_input, truncation=True).to(model.device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_output,
                                 num_beams=4, length_penalty=2.0,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text_mbart50(text, model, tokenizer, max_input, max_output):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=max_input, truncation=True).to(model.device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_output,
                                 num_beams=4, length_penalty=2.0,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text_llama(text, model, tokenizer, max_input, max_output):
    prompt = f"Summarize the following text:\n{text}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", 
                       max_length=max_input, truncation=True, padding=True).to(model.device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output,
        temperature=0.7,  
        top_p=0.9,         
        num_beams=4,
        length_penalty=2.0, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def experiments(model_name, experiment_type):
    """Runs an experiment with the given model and dataset."""
    max_input, max_output = get_max_lengths(model_name)
    print(f"Starting Zero-Shot experiment: on {model_name}")
    print(f"Settings: max_input={max_input}, max_output={max_output}")

    train = pd.read_csv("datasets/train.csv")
    train_fr = pd.read_csv("datasets/train_fr.csv")
    train_cross = pd.read_csv("datasets/train_cross.csv")
    val = pd.read_csv("datasets/val.csv")
    val_fr = pd.read_csv("datasets/val_fr.csv")
    val_cross = pd.read_csv("datasets/val_cross.csv")
    test = pd.read_csv("datasets/test.csv")
    test_fr = pd.read_csv("datasets/test_fr.csv")
    test_cross = pd.read_csv("datasets/test_cross.csv")

    model, tokenizer = download_model(model_name)
    print(f"Model {model_name} loaded successfully.")

    if model_name == "mT5":
        summarize_text = summarize_text_mt5
    elif model_name == "mBART50":
        summarize_text = summarize_text_mbart50
    else:
        summarize_text = summarize_text_llama

    # Call the appropriate function based on experiment type
    if experiment_type == "zero-shot":
        run_zero_shot(model, tokenizer, summarize_text, max_input, max_output, test, test_fr, test_cross)
    elif experiment_type == "1-shot":
        run_1_shot(model, tokenizer, summarize_text, max_input, max_output, train, train_fr, train_cross, test, test_fr, test_cross)
    elif experiment_type == "2-shot":
        run_2_shot(model, tokenizer, summarize_text, max_input, max_output, train, train_fr, train_cross, test, test_fr, test_cross)
    else:
        raise ValueError("Invalid experiment type.")

def run_zero_shot(model, tokenizer, summarize_text, max_input, max_output, test, test_fr, test_cross):
    print("Running Zero-Shot Evaluation...")
    for dataset, name in [(test, "English"), (test_fr, "French"), (test_cross, "Cross-lingual")]:
        generated_summaries = [summarize_text(f"summarize: {row['source']}", model, tokenizer, max_input, max_output) for _, row in dataset.iterrows()]
        reference_summaries = dataset["target"].tolist()
        rouge_scores = compute_rouge_scores(generated_summaries, reference_summaries)
        print(f"{name} ROUGE scores:", rouge_scores)


def run_1_shot(model, tokenizer, summarize_text, max_input, max_output, train, train_fr, train_cross, test, test_fr, test_cross):
    print("Running 1-Shot Evaluation...")
    for dataset, train_data, name in [(test, train, "English"), (test_fr, train_fr, "French"), (test_cross, train_cross, "Cross-lingual")]:
        if name == "English":
            tokenizer.src_lang = "en_XX"
        else:
            tokenizer.src_lang = "fr_XX"
        generated_summaries = []
        reference_summaries = []
        for _, sample in dataset.iterrows():
            one_shot = train_data.sample(1)
            prompt = f"summarize: {one_shot['source'].values[0]}\n{one_shot['target'].values[0]}\n\nsummarize: {sample['source']}"
            generated_summaries.append(summarize_text(prompt, model, tokenizer, max_input, max_output))
            reference_summaries.append(sample["target"])
        rouge_scores = compute_rouge_scores(generated_summaries, reference_summaries)
        print(f"{name} ROUGE scores:", rouge_scores)

def run_2_shot(model, tokenizer, summarize_text, max_input, max_output, train, train_fr, train_cross, test, test_fr, test_cross):
    print("Running 2-Shot Evaluation...")
    for dataset, train_data, name in [(test, train, "English"), (test_fr, train_fr, "French"), (test_cross, train_cross, "Cross-lingual")]:
        if name == "English":
            tokenizer.src_lang = "en_XX"
        else:
            tokenizer.src_lang = "fr_XX"
        generated_summaries = []
        reference_summaries = []
        for _, sample in dataset.iterrows():
            two_shots = train_data.sample(2)
            two_shot1, two_shot2 = two_shots.iloc[0], two_shots.iloc[1]
            prompt = f"summarize: {two_shot1['source'].values[0]}\n{two_shot1['target'].values[0]}\n\nsummarize: {two_shot2['source'].values[0]}\n{two_shot2['target'].values[0]}\n\nsummarize: {sample['source']}"
            generated_summaries.append(summarize_text(prompt, model, tokenizer, max_input, max_output))
            reference_summaries.append(sample["target"])
        rouge_scores = compute_rouge_scores(generated_summaries, reference_summaries)
        print(f"{name} ROUGE scores:", rouge_scores)

def fine_tune(model, tokenizer, summarize_text, max_input, max_output, train, val, test):
    pass

def save_rouge_scores(scores, model_name, experiment_type, dataset_name):
    with open("rouge_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, experiment_type, dataset_name, scores["ROUGE-1"], scores["ROUGE-2"], scores["ROUGE-L"]])
        
def main():
    parser = argparse.ArgumentParser(description="Run experiments with different models.")
    parser.add_argument("--model", type=str, required=True, choices=MODELS.values(), help="The model to use.")
    parser.add_argument("--experiment", type=str, required=True, choices=sum(EXPERIMENTS.values(), []), help="The experiment to run.")
    args = parser.parse_args()
    
    save_rouge_scores(rouge_scores, model_name, "zero-shot", name)

    experiments(args.model, args.experiment)

if __name__ == "__main__":
    main()