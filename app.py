import os
import re
import requests
import subprocess
import sys
import tempfile

from flask import Flask, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)


@app.route('/ocr', methods=['POST'])
def ocr():
    image_url = request.json.get('image_url')
    if not image_url:
        return jsonify({'error': 'Missing image_url parameter'}), 400

    response = requests.get(image_url)

    # 检查请求是否成功
    if response.status_code == 200:

        # 从URL中解析出文件名，进而得到文件扩展名
        parsed_url = urlparse(image_url)
        _, file_ext = os.path.splitext(parsed_url.path)

        # 确保文件扩展名以"."开始，且不为空。如果为空，默认为.jpg
        if not file_ext.startswith('.'):
            file_ext = '.jpg'  # 默认后缀，如果无法从URL中获取扩展名

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(response.content)
            image_path = temp_file.name

        # 这里调整为你的 OCR 脚本所在路径
        ocr_script_path = os.path.join(os.getcwd(), 'tools/infer/predict_system.py')

        # 构建命令
        cmd = [
            sys.executable, ocr_script_path,
            '--image_dir={}'.format(image_path),
            '--det_model_dir=inference/ch_PP-OCRv3_det_infer/',
            '--rec_model_dir=inference/ch_PP-OCRv3_rec_infer/',
            '--cls_model_dir=inference/ch_ppocr_mobile_v2.0_cls_infer/',
            '--use_angle_cls=True', '--use_space_char=True'
        ]

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        ocr_output = result.stdout

        # 删除临时文件
        os.remove(image_path)

        # 定义正则表达式，用于提取dt_boxes数量和处理时间
        dt_boxes_pattern = re.compile(r"dt_boxes num : (\d+), elapsed : (\d+\.\d+)")

        # 定义正则表达式，用于提取预测时间和文件路径
        predict_time_pattern = re.compile(r"Predict time of (.*?): (\d+\.\d+)s")

        # 定义正则表达式，用于提取识别结果和置信度
        # 这里假设识别结果行的格式是[日期] ppocr DEBUG: 识别内容, 置信度
        result_pattern = re.compile(r"\[\d+/\d+/\d+ \d+:\d+:\d+\] ppocr DEBUG: (.+), (\d+\.\d+)")

        dt_boxes_num = {}

        # 使用正则表达式提取dt_boxes数量和处理时间
        dt_boxes_match = dt_boxes_pattern.search(ocr_output)
        if dt_boxes_match:
            dt_boxes_num = dt_boxes_match.group(1)  # 检测到的文本框数量
            elapsed_time = dt_boxes_match.group(2)  # 处理时间
            print(f"dt_boxes num: {dt_boxes_num}, elapsed time: {elapsed_time}s")
        else:
            print("No dt_boxes information found in OCR output.")

        # 初始化变量
        predict_time = {}

        # 检查是否有预测时间匹配项
        predict_time_match = predict_time_pattern.search(ocr_output)
        if predict_time_match:
            predict_time = {
                "image_path": predict_time_match.group(1),
                "time_seconds": predict_time_match.group(2)
            }

        # 提取识别结果
        results = result_pattern.findall(ocr_output)
        results_list = [{"text": result[0], "confidence": result[1]} for result in results]

        # 构建JSON对象
        ocr_json = {
            "boxes_num": dt_boxes_num,
            "predict_time": predict_time,
            "results": results_list
        }

        # 返回JSON响应
        return jsonify(ocr_json)
    else:
        print("Failed to download file")
        return jsonify({'error': 'Failed to download file'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4101)
