ablation : 消融操作的相关代码，part1 - part4 分别为 gate ， Tattetion， Sattention , TemporalAttention

update ：改进相关的代码，new1 - new7 分别对应 使用CNN，对全市场特征m进行Tattention， 使用LSTM， 只使用最后一个时间步， 权重衰减， 多尺度聚合， 调整norm1位置  七个优化方法。 fianl1 使用了多尺度聚合并调整norm1位置，针对csi800数据集，结果有较大的改进。final2调整了norm1位置并只使用最后一个时间步，针对csi300数据集，结果有较大改进。
