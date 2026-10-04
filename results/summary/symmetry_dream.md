## Dream-org/Dream-v0-Instruct-7B: all items vs length-symmetric items
| cfg | n | set_acc | ccer | n sym | set_acc sym | ccer sym |
|---|---|---|---|---|---|---|
| confidence_k16_tnone_bfull_T0.0 | 400 | 0.845 | 0.020 | 213 | 0.817 | 0.028 |
| confidence_k1_t0.9_bfull_T0.0 | 400 | 0.890 | 0.013 | 213 | 0.854 | 0.019 |
| confidence_k1_tnone_bfull_T0.0 | 400 | 0.897 | 0.013 | 213 | 0.869 | 0.019 |
| confidence_k2_tnone_bfull_T0.0 | 400 | 0.880 | 0.013 | 213 | 0.854 | 0.019 |
| confidence_k4_tnone_bfull_T0.0 | 400 | 0.870 | 0.015 | 213 | 0.840 | 0.023 |
| confidence_k8_tnone_bfull_T0.0 | 400 | 0.863 | 0.018 | 213 | 0.826 | 0.028 |
| left_to_right_k1_tnone_bfull_T0.0 | 400 | 0.892 | 0.010 | 213 | 0.864 | 0.014 |

## Dream-org/Dream-v0-Instruct-7B: symmetric BFCL parallel items, slots filled in mention order
| cfg | records (syntax ok, all calls matched) | identity order | correct | all symmetric parallel items | identity order over all |
|---|---|---|---|---|---|
| confidence_k16_tnone_bfull_T0.0 | 77 | 0.961 | 0.883 | 84 | 0.881 |
| confidence_k1_t0.9_bfull_T0.0 | 81 | 0.951 | 0.889 | 84 | 0.917 |
| confidence_k1_tnone_bfull_T0.0 | 83 | 0.952 | 0.892 | 84 | 0.940 |
| confidence_k2_tnone_bfull_T0.0 | 83 | 0.952 | 0.867 | 84 | 0.940 |
| confidence_k4_tnone_bfull_T0.0 | 80 | 0.950 | 0.875 | 84 | 0.905 |
| confidence_k8_tnone_bfull_T0.0 | 77 | 0.961 | 0.883 | 84 | 0.881 |
| left_to_right_k1_tnone_bfull_T0.0 | 80 | 0.975 | 0.900 | 84 | 0.929 |
