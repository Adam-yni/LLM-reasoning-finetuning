import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, StoppingCriteriaList, StoppingCriteria, get_cosine_schedule_with_warmup
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from peft import LoraConfig
import wandb
from liger_kernel.transformers import apply_liger_kernel_to_llama
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure and run a PPO training process.")
    parser.add_argument('--model_name', type=str, required=True, help="The model name or path to be used for training.")
    parser.add_argument('--reward_model_name', type=str, required=True, help="The model name or path to be used for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-6, help="Learning rate for the PPO optimizer.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.") 
    parser.add_argument('--mini_batch_size', type=int, default=1, help="Mini Batch size for training.")
    parser.add_argument('--ppo_epochs', type=int, default=1, help="Number of PPO epochs.")
    parser.add_argument('--log_with', type=str, default="wandb", help="Logging tool to use, e.g., 'wandb' or 'tensorboard'.")
    parser.add_argument('--use_wandb', action='store_true', help="Enable logging with Weights & Biases.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument('--cliprange', type=float, default=0.2, help="Clipping range for PPO updates.")
    parser.add_argument('--cliprange_value', type=float, default=0.2, help="Clipping range for PPO updates.")
    parser.add_argument('--target', type=float, default=1.0, help="Target value for KL.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for random number generation.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the model after training.")
    parser.add_argument('--number_samples', type=int, default=10000, help="Number of samples to be used for training, default=10k.")
    return parser.parse_args()





def main():
    args = parse_arguments()
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    model_name=args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
    tokenizer.pad_token = tokenizer.eos_token

    reward_model_name =args.reward_model_name
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, low_cpu_mem_usage=True,torch_dtype=torch.bfloat16, device_map={"": accelerator.process_index},  load_in_8bit=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name,low_cpu_mem_usage=True)
    reward_tokenizer.padding_side='left'


    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops=[], encounters=1, prompt_length=0):
            super().__init__()
            self.stops = stops
            self.ENCOUNTERS = encounters
            self.prompt_length = prompt_length 

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            generated_ids = input_ids[0][self.prompt_length:] 

            stop_count = sum((stop.to(input_ids.device) == generated_ids).sum().item() for stop in self.stops) 

            return stop_count >= self.ENCOUNTERS


    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in ["\n"]]

    ppo_config = PPOConfig(
        model_name=args.model_name,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        log_with=args.log_with if args.use_wandb else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        cliprange_value=args.cliprange_value,
        cliprange=args.cliprange,
        seed=args.seed,
        whiten_rewards=True,
        lam=0.95,
        gamma=1.0,
        kl_penalty='kl',
        init_kl_coef=0.25,
        target_kl=1.0, #not really usefull, the one to use is target
        target =args.target,
        optimize_cuda_cache=True,
    )


    lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"]
        )


    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,device_map={"": accelerator.process_index},  low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,is_trainable=True,peft_config=lora_config,attn_implementation="flash_attention_2")

    optimizer = AdamW(
            params=model.parameters(),
            weight_decay=0.01,
            lr=ppo_config.learning_rate,
        )

    apply_liger_kernel_to_llama(fused_linear_cross_entropy=True)


    reward_model.eval()
    model.train()  


    torch.backends.cuda.matmul.allow_tf32 = True  # Enable for better stability with bfloat16
    def calculate_rewards_parallel(questions, responses): #VERY IMPORTANT, SMALL CHANGES HERE LEAD TO BIG VARIATIONS FOR ACCURACY, STABILITY
        
        inputs = reward_tokenizer.batch_encode_plus(
            [q + "<next>" + r.strip() for q, r in zip(questions, responses)]
            return_tensors="pt", padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        
        all_rewards = []
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            batch_rewards = torch.sigmoid(outputs.logits).cpu().tolist()
        
        
        for reward, response in zip(batch_rewards, responses):
            reward=reward[0]
            if len(response) > 5:
                if "\begin{align" in response or "sorry" in response or "begin{align" in response or "EA−2ΩBr"  in response or '?' in response or "error" in response or "</br>" in response:
                    reward -= 0.9
                if 'answer' in response and not any(c.isdigit() for c in response):
                    
                    reward -= 0.2
                    
                if len(response) > 30 and reward >0.5: #you may want to try to play on these parameters as they have shown to have great influence on the final accuracy
                    reward *= (1+len(response)/100)
        
                all_rewards.append(reward)
            else:
                all_rewards.append(-1) 
        
        return all_rewards

    def build_dataset(tokenizer, dataset_name):
        train_dataset = load_dataset(dataset_name,split='train_1M', streaming=True) 
        examples = list(train_dataset)
        train_dataset = Dataset.from_list(examples)
        
        def filter_aug_math(example):
            return example["problem_source"] == "augmented_math"
        
        train_dataset = train_dataset.filter(filter_aug_math)
        train_dataset=train_dataset.select([i for i in range(args.number_samples)])
        original_columns = train_dataset.column_names
        num_proc = 1

        def preprocess_function(examples): 
            new_examples = {"query": [], "input_ids": []}
            for question,answer in zip(examples["problem"],examples['generated_solution']):
                list_sentences = answer.split('\n')[:-1]
                list_sentences = [step for step in list_sentences if len(step)>1] #to have real steps and not /n/n
                
                for i in range(1,len(list_sentences)-1):
                    
                    if i == 0:
                        beginning=""
                    else:
                        beginning=' \n '.join(list_sentences[:i])
                        beginning=beginning.strip()

                    query = question 
                    messages = [
                    {"role": "system", "content": "You're an obediant mathematical assistant, You will answer math questions and you must start a new line after each single step."},
                    {"role": "user", "content": "You will answer math questions and you must start a new line after each single step."+query},
                    {"role": "assistant", "content":beginning+" \n "},
                ]
                    tokenized_question = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=False,
                    )
                    new_examples["query"].append(query+beginning)
                    new_examples["input_ids"].append(tokenized_question)
            return new_examples

        ds = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
        ds = ds.filter(lambda x: len(x["input_ids"]) < 2000, batched=False) #avoid OOM 
        ds.set_format(type="torch")
        return ds

    dataset = build_dataset(tokenizer,'nvidia/OpenMathInstruct-2') #built for this dataset

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        config=ppo_config,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    total_steps = len(ppo_trainer.dataloader) * ppo_config.ppo_epochs

    warmup_steps = 0 
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps= warmup_steps, num_training_steps=total_steps)
    ppo_trainer.lr_scheduler=scheduler


    generation_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead 
        "eos_token_id": -1,
        "max_new_tokens": 50, # specify how many tokens you want to generate at most
        "length_penalty": 8.0, # you don't want the model to generate less tokens
    }



    for epoch in range(ppo_config.ppo_epochs):
        print(f"Époque {epoch+1} sur {ppo_config.ppo_epochs}")

        for batch_idx, batch in tqdm(enumerate(ppo_trainer.dataloader), desc=f"Epoch {epoch+1}"):
            accelerator.print(f"Traitement du batch {batch_idx} de l'epoque {epoch+1}")

            question_tensors = [torch.tensor(input_question['input_ids'], dtype=torch.long).squeeze() for input_question in batch["input_ids"]] 
            
            stopping_criteria_list = [
                StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=1, prompt_length=len(q_tensor))])
                for q_tensor in question_tensors
            ]

            with torch.no_grad():
                response_tensors = [
                    ppo_trainer.generate(q_tensor, return_prompt=False, stopping_criteria=sc, **generation_kwargs)[0]
                    for q_tensor, sc in zip(question_tensors, stopping_criteria_list)
                ]
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
            rewards = calculate_rewards_parallel(batch["query"], batch["response"])
            rewards = [torch.tensor(r, dtype=torch.float, device=device) for r in rewards]
        
            loss = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(loss, batch, rewards)

    ppo_trainer.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    wandb.finish()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()