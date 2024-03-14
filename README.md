# OCR-Server

## OCR命令示例
### windows
- 文字位置检测
> tools/infer/predict_det.py --det_model_dir=inference/ch_PP-OCRv3_det_infer/  --image_dir=resources/imgs/11.jpg  --use_gpu=True
- 文本识别 
> tools/infer/predict_system.py --image_dir="resources/imgs/11.jpg" --det_model_dir="inference/ch_PP-OCRv3_det_infer/"  --rec_model_dir="inference/ch_PP-OCRv3_rec_infer/" --cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True
### linux:
> python3 tools/infer/predict_det.py --det_model_dir=inference/ch_PP-OCRv3_det_infer/  --image_dir=resources/imgs/11.jpg  --use_gpu=True

## 安装PaddlePaddle GPU版本
> python -m pip install paddlepaddle-gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple