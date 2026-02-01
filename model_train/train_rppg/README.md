# RPPG
## 使用仓库
~~~
git clone https://github.com/ubicomplab/rPPG-Toolbox.git
cd rPPG-Toolbox
# 需要修改参数
DATA_PATH: 


~~~
## 训练
参数修改
configs/train_configs/PURE_PURE_UBFC-PHYS_EFFICIENTPHYS.yaml

对其中的 TRAIN、TEST进行的PURE数据路径进行修改即可


DATA_PATH: "/PURE/RawData"   

CACHED_PATH: "./xxx/xxx"

~~~
python main.py --config_file PURE_PURE_UBFC-rPPG_EFFICIENTPHYS.yaml
~~~

## 导出模型
在[export_onnx.py](export_onnx.py)文件中修改

pth_path 与 onnx_path 路径即可


