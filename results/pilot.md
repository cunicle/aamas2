| model | mode | cfg | n | bfcl_acc | set_acc | syntax | ccer | scer | nfe |
|---|---|---|---|---|---|---|---|---|---|
| Dream-org/Dream-v0-Instruct-7B | skeleton | confidence_k16_tnone_bfull_T0.0 | 80 | 0.800 | 0.800 | 0.912 | 0.025 | 0.100 | 2.212 |
| Dream-org/Dream-v0-Instruct-7B | skeleton | confidence_k1_tnone_bfull_T0.0 | 80 | 0.850 | 0.850 | 0.975 | 0.013 | 0.125 | 28.350 |
| Dream-org/Dream-v0-Instruct-7B | skeleton | confidence_k4_tnone_bfull_T0.0 | 80 | 0.825 | 0.825 | 0.950 | 0.013 | 0.125 | 7.438 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k16_tnone_b32_T0.0 | 80 | 0.750 | 0.750 | 0.887 | 0.062 | 0.100 | 4.000 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k16_tnone_bfull_T0.0 | 80 | 0.537 | 0.537 | 0.812 | 0.150 | 0.175 | 2.188 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k1_tnone_b32_T0.0 | 80 | 0.850 | 0.850 | 0.938 | 0.013 | 0.075 | 28.225 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k1_tnone_bfull_T0.0 | 80 | 0.637 | 0.637 | 0.875 | 0.138 | 0.138 | 27.425 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k4_tnone_b32_T0.0 | 80 | 0.850 | 0.850 | 0.938 | 0.037 | 0.062 | 8.512 |
| inclusionAI/LLaDA2.0-mini | skeleton | confidence_k4_tnone_bfull_T0.0 | 80 | 0.600 | 0.600 | 0.838 | 0.138 | 0.188 | 7.050 |
dream: 80 runs, 53 ms per forward at mean canvas 527 tokens  ->  --timing dream=73:15
llada2: 80 runs, 536 ms per forward at mean canvas 491 tokens  ->  --timing llada2=1062:15
