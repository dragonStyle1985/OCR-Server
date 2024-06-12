# OCR-Server

## 构建镜像
> docker build -t ocr-server .<br>
> docker save -o ocr-server.tar ocr-server

## CUDA安装测试
> import paddle<br>
  paddle.utils.run_check()

## OCR命令示例
### windows
- 文字位置检测
> tools/infer/predict_det.py --det_model_dir=inference/ch_PP-OCRv3_det_infer/  --image_dir=resources/imgs/11.jpg  --use_gpu=True
- 检测 + 识别
> tools/infer/predict_system.py --image_dir="resources/imgs/11.jpg" --det_model_dir="inference/ch_PP-OCRv3_det_infer/"  --rec_model_dir="inference/ch_PP-OCRv3_rec_infer/" --cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True
### linux:
> python3 tools/infer/predict_det.py --det_model_dir=inference/ch_PP-OCRv3_det_infer/  --image_dir=resources/imgs/11.jpg  --use_gpu=True

## 安装PaddlePaddle GPU版本
### GPU Compute Capability: 8.6
> python -m pip install paddlepaddle-gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

### GPU Compute Capability: 6.1
> python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
