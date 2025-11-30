import os
import sys
import random
from pathlib import Path
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import fire
from datasets import load_dataset, load_from_disk
from qtorch.quant import quantizer
from qtorch import FixedPoint
import pickle
import logging
import datetime

logger = logging.getLogger()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import List, Optional
from models.generation import Llama
from api.datatypes import RawContent, RawMessage, StopReason, ToolPromptFormat
from api.chat_format import ChatFormat, LLMInput
from models.generation import Llama, ChatFormat
from api.datatypes import RawMessage, StopReason, Role
import pandas as pd
import re


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: Optional[List[float]] = None


@dataclass
class ChatPrediction:
    generation: RawMessage
    tokens: Optional[List[int]] = None
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None


choices = ["A", "B", "C", "D"]


def extract_answer_label(text):
    match = re.search(r'\b([A-D])\b', text.upper())
    return match.group(1) if match else None


def append_new_question(dialogs, question, choice_a, choice_b, choice_c, choice_d):
    new_question_dialog = [
        RawMessage(
            role=Role.user.value,
            content=f"""Question: {question}
Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer: """
        )
    ]
    return dialogs + new_question_dialog


def load_few_shot_examples(dev_data_path, num_examples=5):
    df = pd.read_csv(dev_data_path, header=None)
    few_shot_examples = []
    for i in range(min(num_examples, len(df))):
        row = df.iloc[i]
        question, choice_a, choice_b, choice_c, choice_d, answer = row
        few_shot_examples.append(
            RawMessage(
                role=Role.user.value,
                content=f"""Question: {question}
Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer: """
            )
        )
        few_shot_examples.append(
            RawMessage(
                role=Role.assistant.value,
                content=f"{answer}. {df.iloc[i, ord(answer) - ord('A') + 1]}",
                stop_reason=StopReason.end_of_turn
            )
        )
    return few_shot_examples
    
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_perplexity(
        generator: Llama,
        tokenizer,
        dataset,
        max_seq_len: int,
        batch_size: int,
        num_samples: int = 256,
        analyzer: torch.Tensor = None,
        test_data_path: str = "data/dataset/MMLU/test"
):
    device = next(generator.model.parameters()).device
    set_seed(42)
    texts = [sample['text'] for sample in dataset]
    full_text = '\n'.join(texts)
    model_input = generator.formatter.encode_content(full_text)
    all_tokens = model_input.tokens
    input_sequences = [
        all_tokens[i:i + max_seq_len]
        for i in range(0, len(all_tokens), max_seq_len)
        if len(all_tokens[i:i + max_seq_len]) >= 2
    ]
    if not input_sequences:
        return float('inf')
    sampled_sequences = random.sample(input_sequences, min(num_samples, len(input_sequences)))
    all_perplexities = []
    for i in range(0, len(sampled_sequences), batch_size):
        batch_sequences = sampled_sequences[i:i + batch_size]
        batch_input_ids = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_sequences]
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_id)
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        if inputs.size(1) == 0:
            continue
        with torch.no_grad():
            logits = generator.model(inputs, start_pos=0, __ANALYSIS__=analyzer)
            log_probs = torch.log_softmax(logits, dim=-1)
        try:
            log_probs_selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        except Exception as e:
            logger.error(f"Gather operation failed in batch {i}: {e}")
            continue
        valid_mask = (targets != tokenizer.pad_id)
        log_probs_valid = log_probs_selected * valid_mask.float()
        token_counts = valid_mask.sum(dim=1)
        if (token_counts == 0).any():
            log_prob_sum = log_probs_valid.sum(dim=1)[token_counts != 0]
            token_counts = token_counts[token_counts != 0]
        else:
            log_prob_sum = log_probs_valid.sum(dim=1)
        batch_perplexities = torch.exp(-log_prob_sum / token_counts)
        all_perplexities.append(batch_perplexities)
    if not all_perplexities:
        return float('inf')
    overall_perplexity = torch.cat(all_perplexities).mean().item()
    file_name = 'anatomy_test.csv'
    file_path = os.path.join(test_data_path, file_name)
    dev_data_path = file_path.replace("test", "dev")
    few_shot = load_few_shot_examples(dev_data_path)
    df = pd.read_csv(file_path, header=None)
    total_questions = 100
    correct_answers = 0
    correct_answers_list = ''
    predicted_answers_list = ''
    for index in range(0, total_questions, batch_size):
        dialogs = []
        end_idx = min(index + batch_size, total_questions)
        batch = df.iloc[index:end_idx]
        for _, row in batch.iterrows():
            question, choice_a, choice_b, choice_c, choice_d, _ = row
            dialog = append_new_question(few_shot, question, choice_a, choice_b, choice_c, choice_d)
            dialogs.append(dialog)
        results = []
        for dialog in dialogs:
            result = generator.chat_completion(
                dialog,
                max_gen_len=1,
                temperature=0,
                top_p=0.9,
                analyzer=analyzer,
            )
            results.append(result)
        for idx, (dialog, result) in enumerate(zip(dialogs, results)):
            out_message = result.generation.content.strip()
            predicted_token = extract_answer_label(out_message)
            actual_idx = index + idx
            correct_label = extract_answer_label(df.iloc[actual_idx, df.shape[1] - 1])
            if predicted_token is not None:
                predicted_answers_list += predicted_token
            correct_answers_list += correct_label
            if predicted_token == correct_label:
                correct_answers += 1
    accuracy = correct_answers / total_questions * 100
    return overall_perplexity, None, accuracy


def run_main(
        ckpt_dir: str = "/home/iclab/.llama/checkpoints/Llama3.2-1B-Instruct",
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        num_samples: int = 16,
):
    if num_samples == 1:
        base_perplexity = 15.31936455
    elif num_samples == 16:
        base_perplexity = 12.48989868
    elif num_samples == 32:
        base_perplexity = 12.40525818
    print(f'max_batch_size: {max_batch_size}')
    print(f'num_samples: {num_samples}')
    current_device = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    print(f'현재 CUDA device가 {current_device}가 맞습니까? (y/n) : ')
    flag = input()
    if flag != 'y':
        print('CUDA device를 다시 지정하십시오.')
        return
    cuda_num = str(current_device)
    tokenizer_path = str(Path(__file__).parent.parent / "api/tokenizer.model")
    simulation_start = datetime.datetime.now()
    file_name_time = simulation_start.strftime("%Y%m%d_%H%M%S")
    if logger.hasHandlers():
        logger.handlers.clear()
    os.makedirs(f'results/logs-1B-{cuda_num}', exist_ok=True)
    log_file_path = f'results/logs-1B-{cuda_num}/{file_name_time}.log'
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    tokenizer = generator.tokenizer
    save_path = f'data/analyzer_1B/analyzer_1B-{cuda_num}.pt'
    new_path = f'data/analyzer_1B_mod/analyzer_1B-{cuda_num}.pt'
    dataset_path = "data/dataset/wikitext"
    dataset = load_from_disk(dataset_path)
    analyzer = torch.load(save_path, weights_only=True)
    start_node = 0
    base_mmlu = 50
    perplexity = 999
    end_flag = 32
    all_start = time.time()
    logger.info('┌─────────────────────┬─────────────────┬───────────────────────────┬───────────────────────┬────────────────────────┐')
    for i in range(53):
        if analyzer[i, 4] != 99:
            continue
        analyzer[i, 0] = 3
        for bit in range(1, end_flag + 1):
            analyzer[i, 4] = bit
            start = time.time()
            perplexity, sentence, mmlu = calculate_perplexity(
                generator,
                tokenizer,
                dataset,
                max_seq_len,
                max_batch_size,
                num_samples,
                analyzer
            )
            end = time.time()
            scale = 1.006
            torch.save(analyzer, new_path)
            logger.info(
                f"│ Simulation Node: {i:2d} │ Fixed: <{int((analyzer[i, 3]).item()):2d}, "
                f"{int((analyzer[i, 4].item())):2d}> │ ppl: {perplexity:+20.6f} │ "
                f"Chat: {mmlu:15.2f} │ Elapse Time: {end-start:7.2f} s │"
            )
            if (perplexity <= base_perplexity * scale) and (base_mmlu <= mmlu <= base_mmlu + 4):
                analyzer[i, 5] = perplexity
                logger.info('┢━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━━┪')
                logger.info(
                    f"┃ Simulation Node: {i:2d} ┃ Fixed: <{int((analyzer[i, 3]).item()):2d}, "
                    f"{int((analyzer[i, 4].item())):2d}> ┃ ppl: {perplexity:+20.6f} ┃ "
                    f"Chat: {mmlu:15.2f} ┃ Elapse Time: {end-start:7.2f} s ┃"
                )
                logger.info('┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇')
                torch.save(analyzer, new_path)
                break
            elif bit == end_flag:
                analyzer[i, 0] = 2
                analyzer[i, 4] = 99
                analyzer[i, 5] = perplexity
                logger.info('┢━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━╈━━━━━━━━━━━━━━━━━━━━━━━━┪')
                logger.info(
                    f"┃ Simulation Node: {i:2d} ┃ Fixed: <{int((analyzer[i, 3]).item()):2d}, "
                    f"{int((analyzer[i, 4].item())):2d}> ┃ ppl: {perplexity:+20.6f} ┃ "
                    f"Chat: {mmlu:15.2f} ┃ Elapse Time: {end-start:7.2f} s ┃"
                )
                logger.info('┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇')
                torch.save(analyzer, new_path)
    all_end = time.time()
    logger.info(f" Elaspe time: {all_end-all_start:.4} s\n\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
