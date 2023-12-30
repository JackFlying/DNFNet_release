# 安装环境
1. 本代码在CGPS基础上做的改进：https://github.com/ljpadam/CGPS
2. 基于mmdetection框架。mmdet==2.4.0，mmcv-full==1.2.6，pytorch==1.7.0
* ```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch```
* ```pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html```

# 数据集
* For CUHK-SYSU, change the path of your dataset and the annotaion file in the [config file](configs/_base_/datasets/coco_reid_unsup_prw.py) L2, L36, L41, L47, L52
* For PRW, change the path of your dataset and the annotaion file in the [config file](configs/_base_/datasets/coco_reid_unsup_prw.py) L2, L36, L41, L47, L52

# 各种configs介绍
1. configs/cpcl。 论文《Consistent Prototype Contrastive Learning for Weakly Supervised Person Search》
2. configs/dnfnet。 论文《Dual-label Noise Filtering for Weakly Supervised Person Search》
3. configs/dnfnet_siamese_deformable。 毕设的最终版本模型。

# 训练和推理
如果修改了目录下的代码，每次运行前执行 ```python setup.py develop```
1. PRW数据集。 ```cd jobs/prw/```，train.sh和test.sh分别包含训练和测试脚本
2. CUHK-SYSU数据集。```cd jobs/cuhk/```, train.sh和test.sh分别包含训练和测试脚本

# 可视化系统
* 开启后端：```cd tools/system```，```python backend.py```
* 开启前端：前后端链接需要服务器支持，前端代码在tools/system/frontend.zip