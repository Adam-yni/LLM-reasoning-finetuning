import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss
from accelerate import Accelerator
import pandas as pd

accelerator = Accelerator()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure training for PRM")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="./results")
    parser.add_argument('--results_path', type=str, default="./results")
    parser.add_argument('--learning_rate', type=float, default=2e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--run_name', type=str, default="PRM_training")
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()

def find_max_length(dataset, tokenizer):
    max_length = 0
    for example in dataset:
        length = len(tokenizer(example['text'])['input_ids'])
        if length > max_length:
            max_length = length
    return max_length

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)

def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    df = pd.read_excel(args.dataset)
    df = df.dropna(subset=['text', 'labels'])
    df = df.sample(frac=1, random_state=1)
    dataset = Dataset.from_pandas(df)

    train_test_split = dataset.train_test_split(test_size=0.05)
    train_dataset = train_test_split['train']
    validation_dataset = train_test_split['test']

    max_length = 1024 #find_max_length(train_dataset, tokenizer) #or use find max length but this is likely to lead to OOM, better to keep only <1024 length samples
    print(f'Max length is {max_length}')

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length), batched=True
    )
    tokenized_val_dataset = validation_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length), batched=True
    )

    def add_labels(examples):
        return {'labels': [float(label) for label in examples['labels']]}

    tokenized_train_dataset = tokenized_train_dataset.map(add_labels, batched=True)
    tokenized_val_dataset = tokenized_val_dataset.map(add_labels, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    training_args = TrainingArguments(
        output_dir=args.results_path,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        run_name=args.run_name,
        report_to="wandb" if args.use_wandb else None,
        gradient_accumulation_steps=1,
    )

    class CustomTrainer(Trainer): #BCE loss instead of MSE (we want to estimate probabilities between 0 and 1) #this do not work with all versions of transformers
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    accelerator.wait_for_everyone()
    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)




if __name__ == "__main__":
    main()