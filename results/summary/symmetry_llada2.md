## inclusionAI/LLaDA2.0-mini: all items vs length-symmetric items
| cfg | n | set_acc | ccer | n sym | set_acc sym | ccer sym |
|---|---|---|---|---|---|---|
| confidence_k16_tnone_b32_T0.0 | 400 | 0.797 | 0.045 | 216 | 0.806 | 0.019 |
| confidence_k1_tnone_b32_T0.0 | 400 | 0.877 | 0.025 | 216 | 0.875 | 0.009 |
| confidence_k2_tnone_b32_T0.0 | 400 | 0.875 | 0.028 | 216 | 0.880 | 0.009 |
| confidence_k4_tnone_b32_T0.0 | 400 | 0.865 | 0.033 | 216 | 0.861 | 0.014 |
| confidence_k8_tnone_b32_T0.0 | 400 | 0.833 | 0.040 | 216 | 0.838 | 0.014 |

## inclusionAI/LLaDA2.0-mini: symmetric BFCL parallel items, slots filled in mention order
| cfg | records (syntax ok, all calls matched) | identity order | correct |
|---|---|---|---|
| confidence_k16_tnone_b32_T0.0 | 80 | 0.938 | 0.912 |
| confidence_k1_tnone_b32_T0.0 | 82 | 0.951 | 0.963 |
| confidence_k2_tnone_b32_T0.0 | 82 | 0.963 | 0.951 |
| confidence_k4_tnone_b32_T0.0 | 81 | 0.963 | 0.951 |
| confidence_k8_tnone_b32_T0.0 | 81 | 0.951 | 0.938 |
