# 实验结果对比

n=80

**正确** = 识别正确　**未知** = 拒识/高不确定性（安全，触发人工复核）　**误识** = 给出错误药名（**危险**）

| code | 模型 | 知识库 | 引导 | CoT | 正确 | 未知 | 误识 | 误识均NED | 均时 |
|:----:|:----:|:------:|:----:|:---:|:----:|:----:|:----:|:---------:|:----:|
| 0000 | qwen3-vl-8b-instruct | off | off | off | 61.3% | 0.0% | 38.8% | 0.58 | 6.052s |
| 0011 | qwen3-vl-8b-instruct | off | on | on | 61.3% | 0.0% | 38.8% | 0.50 | 18.195s |
| 0101 | qwen3-vl-8b-instruct | on | off | on | 100.0% | 0.0% | 0.0% | - | 21.631s |
| 0110 | qwen3-vl-8b-instruct | on | on | off | 97.5% | 0.0% | 2.5% | 0.56 | 9.251s |
| 0111 | qwen3-vl-8b-instruct | on | on | on | 98.8% | 1.2% | 0.0% | - | 20.549s |
| 1000 | qwen3-vl-8b-instruct-4bit | off | off | off | 62.5% | 0.0% | 37.5% | 0.53 | 4.614s |
| 1011 | qwen3-vl-8b-instruct-4bit | off | on | on | 58.8% | 0.0% | 41.2% | 0.49 | 7.830s |
| 1101 | qwen3-vl-8b-instruct-4bit | on | off | on | 98.8% | 0.0% | 1.2% | 0.44 | 10.917s |
| 1110 | qwen3-vl-8b-instruct-4bit | on | on | off | 97.5% | 0.0% | 2.5% | 0.62 | 7.742s |
| 1111 | qwen3-vl-8b-instruct-4bit | on | on | on | 100.0% | 0.0% | 0.0% | - | 3.537s |
| 2000 | qwen3-vl-4b-instruct | off | off | off | 52.5% | 10.0% | 37.5% | 0.50 | 2.790s |
| 2011 | qwen3-vl-4b-instruct | off | on | on | 53.8% | 2.5% | 43.8% | 0.50 | 7.131s |
| 2101 | qwen3-vl-4b-instruct | on | off | on | 85.0% | 1.2% | 13.8% | 0.58 | 9.903s |
| 2110 | qwen3-vl-4b-instruct | on | on | off | 93.8% | 0.0% | 6.2% | 0.68 | 6.068s |
| 2111 | qwen3-vl-4b-instruct | on | on | on | 87.5% | 0.0% | 12.5% | 0.57 | 4.698s |
| 3000 | qwen2.5-vl-7b-instruct-awq | off | off | off | 63.7% | 0.0% | 36.2% | 0.56 | 5.123s |
| 3011 | qwen2.5-vl-7b-instruct-awq | off | on | on | 56.2% | 6.2% | 37.5% | 0.55 | 18.107s |
| 3101 | qwen2.5-vl-7b-instruct-awq | on | off | on | 55.0% | 15.0% | 30.0% | 0.47 | 14.513s |
| 3110 | qwen2.5-vl-7b-instruct-awq | on | on | off | 86.2% | 0.0% | 13.8% | 0.32 | 6.878s |
| 3111 | qwen2.5-vl-7b-instruct-awq | on | on | on | 75.0% | 7.5% | 17.5% | 0.45 | 17.184s |
