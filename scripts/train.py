from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmForMaskedLM, TrainerCallback
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
import torch
from torch.optim import AdamW
import wandb
import numpy as np
import argparse
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
wandb.login()

class ProteinDataset(Dataset):
    def __init__(self, file, tokenizer):
        data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.proteins = data["Receptor Sequence"].tolist()
        self.peptides = data["Binder"].tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        peptide_seq = self.peptides[idx]

        masked_peptide = '<mask>' * len(peptide_seq)
        complex_seq = protein_seq + masked_peptide

        # Tokenize and pad the complex sequence
        complex_input = self.tokenizer(complex_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)

        input_ids = complex_input["input_ids"].squeeze()
        attention_mask = complex_input["attention_mask"].squeeze()

        # Create labels
        label_seq = protein_seq + peptide_seq
        labels = self.tokenizer(label_seq, return_tensors="pt", padding="max_length", max_length = 552, truncation=True)["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels = torch.where(input_ids == self.tokenizer.mask_token_id, labels, -100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str,required=True, help="please enter wandb entity name")
    parser.add_argument("--wandb_project", type=str,required=True, help="please enter wandb project name")
    parser.add_argument("--train_data", type=str,required=True, help="please enter traning data path")
    parser.add_argument("--test_data", type=str,required=True, help="please enter test data path")
    parser.add_argument("--model_name", default="esm2_t33_650M_UR50D", type=str)
    parser.add_argument("--lr", default=0.0007984276816171436, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--per_device_train_batch_size", default=2, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
    parser.add_argument("--warmup_steps", default=501, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--evaluation_strategy", default="epoch", type=str)
    parser.add_argument("--load_best_model_at_end", default=True, type=bool)
    parser.add_argument("--save_strategy", default="epoch", type=str)
    parser.add_argument("--metric_for_best_model", default="loss", type=str)
    parser.add_argument("--save_total_limit", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    args = parser.parse_args()

    model_name = args.model_name
    model = EsmForMaskedLM.from_pretrained("facebook/" + model_name)
    tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)
    lr = args.lr
    
    training_args = TrainingArguments(
        output_dir='./output/',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        save_strategy=args.save_strategy,
        metric_for_best_model=args.metric_for_best_model,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="wandb"
    )
    

    with wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="training") as run:
        train_dataset = ProteinDataset(args.train_data, tokenizer)
        test_dataset = ProteinDataset(args.test_data, tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(AdamW(model.parameters(), lr=lr), None),
        )
        trainer.train() 

if __name__ == "__main__":
    main()
