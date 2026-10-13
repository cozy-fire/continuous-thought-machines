# 统计口径

{
  "unit": "paired map; one deterministic episode per mode; one model seed",
  "bootstrap_resamples": 10000,
  "bootstrap_seed": 20261004,
  "success_CI": {
    "reset_every_5": [
      0.12427907912587079,
      0.2281588362831245
    ],
    "recurrent": [
      0.11567412031587122,
      0.21714070162066934
    ]
  },
  "recurrent_minus_reset5_success": -0.01,
  "paired_success_difference_CI": [
    -0.04,
    0.015
  ],
  "recurrent_minus_reset5_agreement": -0.22141486500983304,
  "paired_map_bootstrap_agreement_difference_CI": [
    -0.24571999871271816,
    -0.19609019695761568
  ],
  "both_success": 29,
  "reset5_only_success": 5,
  "recurrent_only_success": 3
}

成功率Wilson95%区间；模式差使用配对地图10000次bootstrap，不把决策视作独立样本。不进行训练方法优越性检验。轨迹一致率差受访问分布影响。
