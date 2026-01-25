from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from datetime import datetime
from pathlib import Path
import json

import evaluate
import numpy as np



def main():
    imdb = load_dataset("imdb")
    print(f"imdb [0]", imdb["test"][0])
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    tokenized_imdb = imdb.map(lambda x:preprocess(x, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train(tokenizer, tokenized_imdb, data_collator)
    
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def train(tokenizer, tokenized_imdb, data_collator):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path("runs") / f"classifier_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = 2e-5

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id)
    
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # save config
    (run_dir / "config.json").write_text(json.dumps({
        "lr": learning_rate,
        "pretrained_model_name_or_path": "distilbert/distilbert-base-uncased",
        "training_args": training_args.to_dict()
    }, indent=2))

    trainer.train()





if __name__ == "__main__":
    main()