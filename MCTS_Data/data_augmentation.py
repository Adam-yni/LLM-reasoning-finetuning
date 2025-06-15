import argparse
import pandas as pd
import random
import re
from word2number import w2n
from difflib import SequenceMatcher


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data augmentation script with configurable input/output paths and similarity threshold.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset (Excel file).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the augmented dataset (CSV file).")
    parser.add_argument("--similarity_threshold", type=float, default=0.88, help="Threshold for string similarity (default: 0.88).")
    return parser.parse_args()


def extract_last_result(text):
    matches = re.findall(r'=\s*([\d\+\-\*/\.\s()]+)', text)
    if matches:
        last_calculation = matches[-1]
        try:
            return eval(last_calculation)
        except Exception:
            return None
    return None


def extract_numbers(text):
    return re.findall(r'\d+(?:/\d+)?', text)


def similarity(text1, text2):
    has_equal1 = '=' in text1
    has_equal2 = '=' in text2

    if has_equal1 and has_equal2:
        result1 = extract_last_result(text1)
        result2 = extract_last_result(text2)
        if result1 is not None and result2 is not None:
            if result1 == result2:
                return SequenceMatcher(None, text1, text2).ratio()
            else:
                return 0

    numbers1 = extract_numbers(text1)
    numbers2 = extract_numbers(text2)

    if numbers1 or numbers2:
        last_number1 = numbers1[-1] if numbers1 else None
        last_number2 = numbers2[-1] if numbers2 else None

        if last_number1 and last_number2:
            if last_number1 == last_number2:
                return SequenceMatcher(None, text1, text2).ratio()
            elif len(numbers1) < len(numbers2):
                second_last_number2 = numbers2[-2] if len(numbers2) > 1 else None
                if last_number1 == second_last_number2:
                    return SequenceMatcher(None, text1, text2).ratio()
                else:
                    return 0
            else:
                second_last_number1 = numbers1[-2] if len(numbers1) > 1 else None
                if last_number2 == second_last_number1:
                    return SequenceMatcher(None, text1, text2).ratio()
                else:
                    return 0
        elif last_number1 or last_number2:
            return 0

    return SequenceMatcher(None, text1, text2).ratio()


def split_response(string):
    return re.split('(?<=[.!?])(?=[^0-9])', string)


def words_to_numbers(s):
    words = s.split()
    converted_words = []
    for word in words:
        try:
            converted_words.append(str(w2n.word_to_num(word)))
        except ValueError:
            converted_words.append(word)
    return ' '.join(converted_words)


def false_positive_similarity(s1, s2):
    s1 = words_to_numbers(s1[0])
    s2 = words_to_numbers(s2[0])
    numbers1 = extract_numbers(s1)
    numbers2 = extract_numbers(s2)
    return set(numbers1) == set(numbers2)


def process_list(input_list, MC, threshold):
    counts = {}
    for item in input_list:
        string, value, position, identifiant = str(item[0]), item[3], item[1], item[2]
        if string not in counts:
            counts[string] = {'True': 0, 'False': 0, 'position': position, 'identifiant': identifiant, 'occurrences': []}
        counts[string][str(value)] += 1
        counts[string]['occurrences'].append((position, identifiant, value))

    groups = []
    for string, count in counts.items():
        is_similar = False
        for group in groups:
            if similarity(string, group[0][0]) >= threshold:
                group.append((string, count))
                is_similar = True
                break
        if not is_similar:
            groups.append([(string, count)])

    output_list = []
    for group in groups:
        true_count = sum(count['True'] for _, count in group)
        false_count = sum(count['False'] for _, count in group)
        value = 1 if true_count > 0 and false_count == 0 else 0 if true_count == 0 and false_count > 0 else MC
        for string, count in group:
            for position, identifiant, _ in count['occurrences']:
                output_list.append([string, position, identifiant, value])

    for string, count in counts.items():
        if all(similarity(string, existing_string) < threshold for existing_string, _, _, _ in output_list):
            value = 1 if count['True'] > 0 and count['False'] == 0 else 0 if count['True'] == 0 and count['False'] > 0 else MC
            for position, identifiant, _ in count['occurrences']:
                output_list.append([string, position, identifiant, value])

    return output_list


def process_bad_rollouts(text):
    liste = text.split("), (")[:-1]
    new_liste = ["(" + element + ")" for element in liste]
    new_liste[0] = new_liste[0][2:]
    return [eval(element) for element in new_liste]


def are_valid_parents(identifiant, position, liste_parents):
    if position == 0:
        return True
    exist_parent = False
    for parent in liste_parents:
        if float(parent[3]) == 0.0 and parent[2] == identifiant:
            return False
        elif float(parent[3]) > 0 and parent[2] == identifiant:
            exist_parent = True
    return exist_parent


def is_well_formatted(sentence):
    return sentence.endswith('.') and not re.search(r'\d+\.\d+', sentence)


def convert_sentences(text):
    text = text.replace("""\n""", '')
    sentences = re.split(r'(?<!\d)\. (?!\d)|\s{2,}', text)
    formatted_sentences = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            if is_well_formatted(stripped_sentence):
                formatted_sentences.append(stripped_sentence)
            else:
                formatted_sentences.append(stripped_sentence + '.')
    converted_text = ' '.join(formatted_sentences)
    return re.sub(r'\.\s+\.', '.', converted_text)


def main():
    args = parse_arguments()
    df = pd.read_excel(args.input_path)
    columns = ['question', 'prefix', 'action', 'value']
    new_df = pd.DataFrame(columns=columns)

    for i in df.index:
        if df['mc_estimation'][i] < 0.06:
            continue
        no_positives = False
        liste_actions = []
        liste_parents = []
        identifiant = 0
        rollouts = process_bad_rollouts(df['rollouts'][i]) if df['rollouts'][i][-1] != "]" else eval(df['rollouts'][i])
        mc = df['mc_estimation'][i]
        question = df['question'][i]
        liste_rollouts = []

        if "!" in rollouts[0][0] or no_positives:
            continue

        for roll in rollouts:
            roll = list(roll)
            roll[1] = convert_sentences(roll[1])
            roll = tuple(roll)
            usable = True
            if "!" in roll[:10]:
                usable = False
                continue

            debut, roll, correct = roll
            splitted_roll = split_response(roll)
            new_roll = []
            if "!" not in splitted_roll[0]:
                for element in splitted_roll:
                    if element != splitted_roll[-1]:
                        element.replace("!", "")
                    if element == "":
                        continue
                    if "..." in element or "Q:" in element or "A:" in element:
                        usable = False
                        continue
                    if "!!!" in element:
                        element1 = element.split("!!!")[0]
                        if len(element1) > 20 and "!" not in element1:
                            new_roll.append(element1)
                    elif "!" not in element:
                        if any(char.isdigit() for char in element) and len(element) > 3:
                            new_roll.append(element)
                        elif len(element) > 15:
                            new_roll.append(element)
            else:
                usable = False
            if not usable:
                continue

            position = 0
            splitted_roll = new_roll
            for ro in splitted_roll:
                liste_rollouts.append((ro, position, identifiant, correct))
                position += 1

            identifiant += 1
            max_position = max(liste_rollouts, key=lambda x: x[1])[1] if liste_rollouts else 0
            candidates = []

            for position in range(max_position + 1):
                for morceau in liste_rollouts:
                    if morceau[1] == position:
                        identifiant, position = morceau[2], morceau[1]
                        valid_parents = are_valid_parents(identifiant, position, liste_parents)
                        if valid_parents:
                            candidates.append(morceau)

                output = process_list(candidates, mc, args.similarity_threshold)
                for morceau in output:
                    liste_parents.append(morceau)

                if position == 0:
                    for i in range(len(output)):
                        new_row = pd.DataFrame({'question': [question], 'prefix': [debut], 'action': [output[i][0]], 'value': [output[i][3]]})
                        new_df = pd.concat([new_df, new_row], ignore_index=True)
                else:
                    for action in output:
                        if action[0] not in liste_actions:
                            liste_actions.append(action[0])
                            liste_antecedent = [rolls for rolls in liste_rollouts if rolls[2] == action[2] and rolls[1] < action[1]]
                            liste_antecedent = sorted(liste_antecedent, key=lambda x: x[1])
                            step = "".join(antecedent[0] for antecedent in liste_antecedent)
                            new_row = pd.DataFrame({'question': [question], 'prefix': [debut + step], 'action': [action[0]], 'value': [action[3]]})
                            new_df = pd.concat([new_df, new_row], ignore_index=True)

                candidates = []

    new_df = new_df.drop_duplicates(subset=['prefix', 'action'])
    new_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()