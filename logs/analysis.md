# Ablation Results

n=80, metrics = correct% / unknown% / misid%

| Config | Qwen2.5-VL-7B-Instruct-AWQ | Qwen3-VL-32B-Instruct | Qwen3-VL-4B-Instruct | Qwen3-VL-8B-Instruct-AWQ-4bit | Qwen3-VL-8B-Instruct |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 64 / 0 / 36 | 65 / 0 / 35 | 52 / 10 / 38 | 64 / 0 / 36 | 61 / 0 / 39 |
| kb | 86 / 0 / 14 | 100 / 0 / 0 | 95 / 0 / 5 | 95 / 0 / 5 | 98 / 0 / 2 |
| guide | 56 / 6 / 38 | 66 / 0 / 34 | 54 / 8 / 39 | 61 / 0 / 39 | 60 / 0 / 40 |
| cot | 10 / 76 / 14 | 64 / 0 / 36 | 52 / 11 / 36 | 64 / 0 / 36 | 55 / 1 / 44 |
| kb+guide | 86 / 0 / 14 | 100 / 0 / 0 | 94 / 0 / 6 | 96 / 0 / 4 | 98 / 0 / 2 |
| kb+cot | 55 / 15 / 30 | 100 / 0 / 0 | 86 / 2 / 11 | 99 / 0 / 1 | 100 / 0 / 0 |
| guide+cot | 56 / 6 / 38 | 65 / 0 / 35 | 54 / 2 / 44 | 60 / 0 / 40 | 61 / 0 / 39 |
| kb+guide+cot | 76 / 6 / 18 | 100 / 0 / 0 | 88 / 0 / 12 | 99 / 0 / 1 | 99 / 0 / 1 |

## Counts (correct / unknown / misid)

| Config | Qwen2.5-VL-7B-Instruct-AWQ | Qwen3-VL-32B-Instruct | Qwen3-VL-4B-Instruct | Qwen3-VL-8B-Instruct-AWQ-4bit | Qwen3-VL-8B-Instruct |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 51 / 0 / 29 | 52 / 0 / 28 | 42 / 8 / 30 | 51 / 0 / 29 | 49 / 0 / 31 |
| kb | 69 / 0 / 11 | 80 / 0 / 0 | 76 / 0 / 4 | 76 / 0 / 4 | 78 / 0 / 2 |
| guide | 45 / 5 / 30 | 53 / 0 / 27 | 43 / 6 / 31 | 49 / 0 / 31 | 48 / 0 / 32 |
| cot | 8 / 61 / 11 | 51 / 0 / 29 | 42 / 9 / 29 | 51 / 0 / 29 | 44 / 1 / 35 |
| kb+guide | 69 / 0 / 11 | 80 / 0 / 0 | 75 / 0 / 5 | 77 / 0 / 3 | 78 / 0 / 2 |
| kb+cot | 44 / 12 / 24 | 80 / 0 / 0 | 69 / 2 / 9 | 79 / 0 / 1 | 80 / 0 / 0 |
| guide+cot | 45 / 5 / 30 | 52 / 0 / 28 | 43 / 2 / 35 | 48 / 0 / 32 | 49 / 0 / 31 |
| kb+guide+cot | 61 / 5 / 14 | 80 / 0 / 0 | 70 / 0 / 10 | 79 / 0 / 1 | 79 / 0 / 1 |

## Avg Response Time (s)

| Config | Qwen2.5-VL-7B-Instruct-AWQ | Qwen3-VL-32B-Instruct | Qwen3-VL-4B-Instruct | Qwen3-VL-8B-Instruct-AWQ-4bit | Qwen3-VL-8B-Instruct |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 5.22 | 13.49 | 2.11 | 4.85 | 6.31 |
| kb | 5.59 | 15.25 | 2.62 | 7.66 | 9.10 |
| guide | 5.08 | 13.28 | 1.95 | 4.72 | 6.14 |
| cot | 5.66 | 43.26 | 5.39 | 7.59 | 18.30 |
| kb+guide | 5.64 | 15.40 | 2.67 | 7.85 | 9.30 |
| kb+cot | 9.92 | 44.08 | 6.31 | 10.94 | 21.42 |
| guide+cot | 12.27 | 42.21 | 5.64 | 7.72 | 18.17 |
| kb+guide+cot | 11.91 | 43.00 | 6.05 | 10.57 | 20.13 |
