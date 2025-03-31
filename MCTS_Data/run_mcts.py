import argparse
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from OmegaPRM_mcts import MonteCarloTreeSearch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure MCTS")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--search_limit', type=int, default=12)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--L', type=int, default=500)
    parser.add_argument('--cpuct', type=float, default=0.125)
    parser.add_argument('--output_path', type=str, default='mcts_data')
    parser.add_argument('--number_samples', type=int, default=1000)
    parser.add_argument('--repetition', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Charger le dataset
    dataset = pd.read_csv(args.dataset)

    # Charger le modèle et le tokenizer
    model_path = args.model
    model = LLM(
        model=model_path,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.95,
        dtype='bfloat16'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Paramètres d'échantillonnage
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=396
    )

    # Initialisation de la recherche Monte Carlo
    MCTS = MonteCarloTreeSearch(
        model=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        search_limit=args.search_limit,
        alpha=args.alpha,
        beta=args.beta,
        L=args.L,
        cpuct=args.cpuct
    )
    tree,tree_time= MCTS.create_tree(dataset,args.repetition,args.number_samples)

    tree.to_excel(args.output_path+'.xlsx', engine='xlsxwriter')

    print(f"Tree created in {tree_time}s")


if __name__ == "__main__":
    main()