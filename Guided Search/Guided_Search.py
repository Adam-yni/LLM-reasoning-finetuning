import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import re
import time
from string import punctuation
from sentence_transformers import SentenceTransformer, util
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure and run a PPO training process.")
    parser.add_argument('--model_name', type=str, required=True, help="The model name or path to be used for training.")
    parser.add_argument('--reward_model_name', type=str, required=True, help="The model name or path to be used for training.")
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--number_of_proposed_steps', type=int, required=True) #if ==1, will be greedy decoding
    parser.add_argument('--max_steps', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=159)
    
    return parser.parse_args()

accelerator = Accelerator()

def main():
    args=parse_arguments()
    llm = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": accelerator.process_index})
    prm = AutoModelForSequenceClassification.from_pretrained(args.reward_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": accelerator.process_index})


    examples = ['\n \n']+[f"{p}\n" for p in punctuation] +[f"{p}\n\n" for p in punctuation] +[f"{p}\n\n\n" for p in punctuation] +[f"{p} \n" for p in punctuation] +[f"{p} \n\n" for p in punctuation] +[f"{p} \n\n\n" for p in punctuation] +[f"{p1}{p2}\n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2}\n\n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2}\n\n\n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2} \n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2} \n\n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2} \n\n\n" for p1 in punctuation for p2 in punctuation] +[f"{p1}{p2}{p3}\n" for p1 in punctuation for p2 in punctuation for p3 in punctuation] +[f"{p1}{p2}{p3}\n\n" for p1 in punctuation for p2 in punctuation for p3 in punctuation] +[f"{p1}{p2}{p3}\n\n\n" for p1 in punctuation for p2 in punctuation for p3 in punctuation] +[f"{p1}{p2}{p3} \n" for p1 in punctuation for p2 in punctuation for p3 in punctuation] +[f"{p1}{p2}{p3} \n\n" for p1 in punctuation for p2 in punctuation for p3 in punctuation] +[f"{p1}{p2}{p3} \n\n\n" for p1 in punctuation for p2 in punctuation for p3 in punctuation]


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)

    def get_newline_token_id(tokenizer,examples):
        """
        Get all token IDs associated with '\n'.
        """

        token_ids = set()
        for ex in examples:
            token_ids.update([tokenizer.encode(ex, add_special_tokens=False)[-1]])
        return list(token_ids)

    ending_tokens = get_newline_token_id(tokenizer,examples)
    stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in examples
            ]

    model_sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops=[], encounters=1, prompt_length=0, initial_ignore=10):
            super().__init__()
            self.stops = [stop[-1] for stop in stops]
            self.ENCOUNTERS = encounters
            self.prompt_length = prompt_length
            self.initial_ignore = initial_ignore
            self.encountered_count = 0
            self.processed_tokens = 0

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            generated_ids = input_ids[0][self.prompt_length:]
            total_generated = len(generated_ids)
            if total_generated <= self.initial_ignore:
                self.processed_tokens = total_generated
                return False
            for token_id in generated_ids[self.processed_tokens:]:
                if any((token_id == stop.to(input_ids.device)).all() for stop in self.stops):
                    self.encountered_count += 1
                if self.encountered_count >= self.ENCOUNTERS:
                    return True
            self.processed_tokens = total_generated
            return False

    def rm(question, response_start, reasoning_step):
        inputs = question + " " + response_start + "<next>" + reasoning_step.strip()
        inputs = prm_tokenizer(inputs, return_tensors="pt").to(prm.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            outputs = prm(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        return probabilities.item()

    def extract_boxed_answer(text):
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            try:
                return float(boxed_match.group(1))
            except ValueError:
                return -999999

        number_match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if number_match:
            try:
                return float(number_match[-1])
            except ValueError:
                return -999999

        return -999999

    def extract_boxed_answer_dataset(text):
        match = text.split('####')[1].replace(',','')
        try:
            return float(match)
        except:
            return -999999

    def generate_solution(question, max_steps=1000, k=8):
        ongoing_solution = ""
        L_steps = []
        L_scores = []
        semantic_scores = []

        for _ in range(max_steps):
            messages_initial = [[
                {"role": "system", "content": "You're an obediant mathematical assistant, You will answer math questions and you must start a new line after each single step."},
                {"role": "user", "content": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"},
                {"role": "assistant", "content": "There are 15 trees originally. \n Then there were 21 trees after some more were planted. \n So there must have been 21 - 15 = 6. \n The answer is $\\boxed{6}$."},
                {"role": "user", "content": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"},
                {"role": "assistant", "content": "There are originally 3 cars. \n 2 more cars arrive. \n 3 + 2 = 5. \n The answer is $\\boxed{5}$."},
                {"role": "user", "content": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"},
                {"role": "assistant", "content": "Originally, Leah had 32 chocolates. \n Her sister had 42. \n So in total they had 32 + 42 = 74. \n After eating 35, they had 74 - 35 = 39. \n The answer is $\\boxed{39}$."},
                {"role": "user", "content": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"},
                {"role": "assistant", "content": "Jason started with 20 lollipops. \n Then he had 12 after giving some to Denny. \n So he gave Denny 20 - 12 = 8. \n The answer is $\\boxed{8}$."},
                {"role": "user", "content": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"},
                {"role": "assistant", "content": "Shawn started with 5 toys. \n If he got 2 toys each from his mom and dad, then that is 4 more toys. \n 5 + 4 = 9. \n The answer is $\\boxed{9}$."},
                {"role": "user", "content": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"},
                {"role": "assistant", "content": "There were originally 9 computers. \n For each of 4 days, 5 more computers were added. \n So 5 * 4 = 20 computers were added. \n 9 + 20 is 29. \n The answer is $\\boxed{29}$."},
                {"role": "user", "content": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"},
                {"role": "assistant", "content": "Michael started with 58 golf balls. \n After losing 23 on tuesday, he had 58 - 23 = 35. \n After losing 2 more, he had 35 - 2 = 33 golf balls. \n The answer is $\\boxed{33}$."},
                {"role": "user", "content": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"},
                {"role": "assistant", "content": "Olivia had 23 dollars. \n 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. \n So she has 23 - 15 dollars left. \n 23 - 15 is 8. \n The answer is $\\boxed{8}$."},
                {"role": "user", "content": 'You will answer this question and start a new line after each single step.' + question},
            ]]*k

            messages = [[
                {"role": "system", "content": "You're an obediant mathematical assistant, You will answer math questions and you must start a new line after each single step."},
                {"role": "user", "content": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"},
                {"role": "assistant", "content": "There are 15 trees originally. \n Then there were 21 trees after some more were planted. \n So there must have been 21 - 15 = 6. \n The answer is $\\boxed{6}$."},
                {"role": "user", "content": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"},
                {"role": "assistant", "content": "There are originally 3 cars. \n 2 more cars arrive. \n 3 + 2 = 5. \n The answer is $\\boxed{5}$."},
                {"role": "user", "content": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"},
                {"role": "assistant", "content": "Originally, Leah had 32 chocolates. \n Her sister had 42. \n So in total they had 32 + 42 = 74. \n After eating 35, they had 74 - 35 = 39. \n The answer is $\\boxed{39}$."},
                {"role": "user", "content": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"},
                {"role": "assistant", "content": "Jason started with 20 lollipops. \n Then he had 12 after giving some to Denny. \n So he gave Denny 20 - 12 = 8. \n The answer is $\\boxed{8}$."},
                {"role": "user", "content": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"},
                {"role": "assistant", "content": "Shawn started with 5 toys. \n If he got 2 toys each from his mom and dad, then that is 4 more toys. \n 5 + 4 = 9. \n The answer is $\\boxed{9}$."},
                {"role": "user", "content": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"},
                {"role": "assistant", "content": "There were originally 9 computers. \n For each of 4 days, 5 more computers were added. \n So 5 * 4 = 20 computers were added. \n 9 + 20 is 29. \n The answer is $\\boxed{29}$."},
                {"role": "user", "content": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"},
                {"role": "assistant", "content": "Michael started with 58 golf balls. \n After losing 23 on tuesday, he had 58 - 23 = 35. \n After losing 2 more, he had 35 - 2 = 33 golf balls. \n The answer is $\\boxed{33}$."},
                {"role": "user", "content": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"},
                {"role": "assistant", "content": "Olivia had 23 dollars. \n 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. \n So she has 23 - 15 dollars left. \n 23 - 15 is 8. \n The answer is $\\boxed{8}$."},
                {"role": "user", "content":'You will answer this question and start a new line after each single step, each step and calculus must be done in the same line. You will encapsulate your final resultat within \boxed{final result}. ' + question},
                {"role": "assistant", "content": ongoing_solution + "\n"},
            ]]*k

            if _ == 0:
                tokenized_question = tokenizer.apply_chat_template(
                            messages_initial,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            truncation=False,
                        ).to(llm.device)
            else:
                tokenized_question = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            truncation=False,
                        ).to(llm.device)

            

            stopping_criteria = [StoppingCriteriaList([
                StoppingCriteriaSub(stops=stop_words_ids, encounters=1, prompt_length=len(tokenized_question['input_ids'][i]))
            ]) for i in range(len(tokenized_question['input_ids']))]

            if args.number_of_proposed_steps >1:
                outputs = llm.generate(
                    input_ids=tokenized_question["input_ids"],
                    attention_mask=tokenized_question["attention_mask"],
                    max_new_tokens=200,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            elif args.number_of_proposed_steps == 1:
                outputs = llm.generate(
                        input_ids=tokenized_question["input_ids"],
                        attention_mask=tokenized_question["attention_mask"],
                        max_new_tokens=200,
                        stopping_criteria=stopping_criteria,
                        do_sample=False,
                    )

            
            T = []
            for output in outputs:
                output = output[len(tokenized_question["input_ids"][0]):]
                
                added=False
                for j in range(5,len(output)):
                    if output[j] in ending_tokens:
                        T.append(output[:j+1])
                        added=True
                        break
                if not added:
                    T.append(output)
                        
            outputs = T
            
            candidates = [(tokenizer.decode(output, skip_special_tokens=True).strip(), output) for output in outputs]
            scores = [rm(question, ongoing_solution, step[0]) for step in candidates]
            steps = [seq[0] for seq in candidates]
            
            best_step = candidates[scores.index(max(scores))]
            ongoing_solution += "\n" + best_step[0].strip()

            # Calculate semantic similarity for the current step
            embeddings = model_sentence_transformer.encode(steps, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            for i in range(len(steps)):
                for j in range(i + 1, len(steps)):
                    semantic_scores.append(cosine_scores[i][j].item())

            end = False
            if tokenizer.eos_token_id in best_step[1] or 'boxed' in best_step[0]:
                end = True
                break

        # Calculate the average semantic similarity score for the current response, useful to check for answer diversity
        if semantic_scores:
            average_semantic_score = sum(semantic_scores) / len(semantic_scores)
        else:
            average_semantic_score = 0.0

        accelerator.print(ongoing_solution)
        return ongoing_solution, average_semantic_score

    def evaluate_dataset(dataset, max_steps=args.max_steps, k=args.number_of_proposed_steps):
        results = []
        semantic_scores = []
        for item in dataset:
            question = item["question"]
            expected_answer = extract_boxed_answer_dataset(item["answer"])
            if expected_answer != -999999:
                solution, semantic_score = generate_solution(question, max_steps=max_steps, k=k)
                predicted_answer = extract_boxed_answer(solution)
                results.append({"question": question, "expected": expected_answer, "predicted": predicted_answer, "text": solution})
                semantic_scores.append(semantic_score)
        return results, semantic_scores

    math_hard = load_dataset("lighteval/MATH-Hard", split='train').select(i for i in range(159))
    #math_hard = load_dataset('openai/gsm8k','main', split='test').select(i for i in range(1000))
    def is_valid_example(example):
        return extract_boxed_answer_dataset(example["answer"]) != -999999

    math_hard = math_hard.filter(is_valid_example)

    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(math_hard) as dataset:
        results, semantic_scores = evaluate_dataset(dataset)
    accelerator.wait_for_everyone()
    results_gathered = gather_object(results)
    semantic_scores_gathered = gather_object(semantic_scores)

    if accelerator.is_main_process:
        timediff = time.time() - start
        accuracies = [1 if res["expected"] == res["predicted"] else 0 for res in results_gathered]
        average_accuracy = sum(accuracies) / len(accuracies)
        overall_average_semantic_score = sum(semantic_scores_gathered) / len(semantic_scores_gathered)
        print(f"Average Accuracy: {average_accuracy:.2%}")
        print(f"Overall Average Semantic Similarity Score: {overall_average_semantic_score:.2f}")
        print(f"Time elapsed: {timediff:.2f} seconds")

if __name__ == "__main__":
    main()
#accelerate launch GS_instruct.py