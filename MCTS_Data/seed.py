import argparse
import re
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_arguments():
    parser = argparse.ArgumentParser(description="launch script with model and number of smamples.")
    parser.add_argument('--model', type=str, required=True, help="path to the model.")
    parser.add_argument('--num_samples', type=int, required=True, help="number of samples to use.")
    return parser.parse_args()

def filter_aug_math(example):
    return example["problem_source"] == "augmented_math"

def create_data(k, data, tokenizer):
    data_created = []
    for i in range(k):
        messages = [
            {'role':'system','content':"You're an obediant mathematical assistant, You will answer math questions and start a new line after each step. Your final answer must be encapsulated with \boxed{final answer}. You must rely on preivous answers to format the final result you will give in \boxed{final answer}."},
            {'role':'user','content':"""What is the range of the function $y = \frac{x^2 + 3x + 2}{x+1}$? (Express your answer using interval notation.)"""},
            {'role':'assistant','content':"""We can factor the numerator to get $y = \frac{(x+1)(x+2)}{x+1}$. \n If we exclude the case where $x = -1$, the function is equivalent to $y = x+2$. \n However, because $x$ cannot equal $-1$, $y$ cannot equal 1. \n Therefore, the range is all real numbers except for 1, which we may write as $y \in \boxed{(-\infty, 1)\cup(1, \infty)}.$"""},
            {'role':'user','content':"""Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."""},
            {'role':'assistant','content':"""For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. \n For example, $ax+3$ and $x-5$ must be equal when $x=2$. \n This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. \n Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. \n Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$."""},
            {'role':'user','content':"""Suppose $a$, $b,$ and $c$ are positive numbers satisfying: \begin{align*} a^2/b &= 1, \\ b^2/c &= 2, \text{ and}\\ c^2/a &= 3. \end{align*}"""},
            {'role':'assistant','content':"""Notice that multiplying all three of the original equations together tells us that $(a^2b^2c^2)/(abc) = 6$, which implies $abc=6$. \n Rewriting the first and third equations as $b = a^2$ and $c = \sqrt{3a}$ and plugging these into $abc=6$ yields $a \cdot a^2\cdot \sqrt{3a} = 6$. \n By squaring both sides of the equation, we obtain $3a^7 = 36 \Rightarrow a = \boxed{12^{1/7}}$."""},
            {'role':'user','content':"""An infinite geometric series has a first term of $12$ and a second term of $4.$ A second infinite geometric series has the same first term of $12,$ a second term of $4+n,$ and a sum of four times that of the first series. Find the value of $n.$"""},
            {'role':'assistant','content':"""Note that if the the two series have constant ratios of $a$ and $b,$ respectively, then $4\left( \frac{12}{1-a} \right) = \frac{12}{1-b}.$ \n Simplifying, $4(1-b)=1-a.$ Substituting in $a= \frac{4}{12}=\frac{1}{3}$ and $b= \frac{4+n}{12}=\frac{1}{3}+\frac{n}{12},$ we quickly find that $n=\boxed{6}.$"""},
            {'role':'user','content':data['problem'][i]},
        ]
        data_created.append([
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            data['problem'][i],
            data['generated_solution'][i]
        ])
    return data_created

def main():
    args = parse_arguments()

    train_dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train_1M', streaming=True)
    examples = list(train_dataset)
    train_dataset = Dataset.from_list(examples)
    train_dataset = train_dataset.filter(filter_aug_math)

    train_dataset = train_dataset.select([i for i in range(args.num_samples)])
    data = train_dataset

    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=1024)
    model = LLM(model=model_path, tensor_parallel_size=8, gpu_memory_utilization=0.95, dtype='bfloat16')

    question_dataset = create_data(len(data), data, tokenizer)

    columns = ['question', 'answer', 'ground_truth_answer', 'correctness']
    df = pd.DataFrame(columns=columns)
    prompts_all = [question[0] for question in question_dataset]
    generated_text = []

    outputs = model.generate(prompts_all, sampling_params)
    for output in outputs:
        generated_text.append(output.outputs[0].text)

    for i in range(len(generated_text)):
        try:
            new_row = pd.DataFrame({
                'question': [question_dataset[i][1]],
                'answer': [generated_text[i]],
                'ground_truth_answer': [question_dataset[i][2]],
                'correctness': [re.findall(r'xed{([^}]*)}', generated_text[i])[0].strip() ==
                                re.findall(r'xed{([^}]*)}', question_dataset[i][2])[0].strip()],
            })
            df = pd.concat([df, new_row], ignore_index=True)
        except Exception as e:
            print(f"eror step {i}: {e}")

    # Sauvegarder le DataFrame
    df.to_csv('seed.csv', index=False)

if __name__ == "__main__":
    main()