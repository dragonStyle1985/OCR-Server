import json
import os
import time

import numpy as np
import requests
import subprocess
import tempfile
import tools.infer.utility as utility

from config import HIDDEN_PATH
from docx import Document   # python-docx
from flask import Flask, request, jsonify
from urllib.parse import urlparse

from tools.infer.predict_system import TextSystem
from util import process_images

app = Flask(__name__)


# 解析命令行参数
args = utility.parse_args()

args.det_model_dir = 'inference/ch_PP-OCRv3_det_infer/'
args.rec_model_dir = 'inference/ch_PP-OCRv3_rec_infer/'
args.cls_model_dir = 'inference/ch_ppocr_mobile_v2.0_cls_infer/'
args.use_angle_cls = True
args.use_space_char = True

# 使用解析后的参数
text_sys = TextSystem(args)

# warm up 10 times
if args.warmup:
    img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    for i in range(10):
        res = text_sys(img)


def parse_ocr_results(results):
    ocr_results = []

    for result in results:
        # 分割图片文件名和 JSON 字符串
        if '\t' in result:
            image_name, data_str = result.split('\t', 1)
            try:
                # 加载 JSON 数据
                data = json.loads(data_str)

                # 提取信息并构建结果列表
                items = []
                for item in data:
                    transcription = item['transcription']
                    points = item['points']
                    score = item['score']
                    items.append({"transcription": transcription, "points": points, "score": score})

                # 构建最终的 JSON 对象
                ocr_json = {
                    "image_name": image_name,
                    "results": items
                }
                ocr_results.append(ocr_json)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from result associated with {image_name}")

    return ocr_results


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    测试方法：
    http://127.0.0.1:4101/ocr
    在Body中填入
    {"image_url": "E:/1.jpg"}
    cv2.imread不支持中文路径
    :return:
    """
    print(1, request.json)
    image_url = request.json.get('image_url')

    if not image_url:
        return jsonify({'error': 'Missing image_url parameter'}), 400

    time1 = time.time()

    image_url = image_url.replace(HIDDEN_PATH, "")
    print(2, image_url)
    if os.path.isabs(image_url):
        print(3)
        # 处理本地文件
        if os.path.exists(image_url):
            image_path = image_url
        else:
            return jsonify({'error': 'File not found on server'}), 404
    else:
        print(4)
        response = requests.get(image_url)
        # 从URL中解析出文件名，进而得到文件扩展名
        parsed_url = urlparse(image_url)
        print(5, parsed_url)
        _, file_ext = os.path.splitext(parsed_url.path)

        # 确保文件扩展名以"."开始，且不为空。如果为空，默认为.jpg
        if not file_ext.startswith('.'):
            file_ext = '.jpg'  # 默认后缀，如果无法从URL中获取扩展名

        print(6, response.status_code)

        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(response.content)
                image_path = temp_file.name

                print(7, image_path)
        else:
            return jsonify({'error': 'Failed to download image from URL'}), response.status_code

    try:
        result, predict_time = process_images(
            text_sys,
            image_dir=image_path,
            process_id=0,
            total_process_num=1,
            page_num=10,
        )

        time2 = time.time()
        print(f'OCR指令执行时间： {time2 - time1}秒')

        parsed_results = parse_ocr_results(result)
        results = parsed_results[0]['results']
        ocr_json = {
            "boxes_num": len(results),
            "predict_time": predict_time,
            "results": parsed_results[0]['results']
        }
        return jsonify(ocr_json), 200

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error message: {e.stderr}")
        return jsonify({'error': 'OCR processing failed', 'stderr': e.stderr}), 500


@app.route('/fetch-images-from-document', methods=['POST'])
def save_docx_images():
    content = request.json
    docx_path = content['file_path']
    output_dir = content['output_dir']

    # print('docx_path', docx_path)
    # print('output_dir', output_dir)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response = requests.get(docx_path)
    if response.status_code == 200:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(response.content)
            docx_file_path = temp_file.name
            # print('docx_file_path', docx_file_path)

        # 加载文档
        docx = Document(docx_file_path)

        rels = docx.part.rels
        images_saved = []
        for rel_id, rel in rels.items():
            if "image" in rel.reltype:
                image_blob = rel.target_part.blob
                image_extension = os.path.splitext(rel.target_ref)[1]
                image_filename = f"{rel_id}{image_extension}"
                image_path = os.path.join(output_dir, image_filename)
                # print('image_path', image_path)

                with open(image_path, 'wb') as img_file:
                    img_file.write(image_blob)
                images_saved.append(image_path)

        # 删除临时文件
        os.remove(docx_file_path)
        return jsonify({"message": "Images saved successfully", "images": images_saved})
    else:
        return jsonify({"message": "Failed to download the docx file"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4101)
