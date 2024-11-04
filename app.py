import json
import os
import sys
import time

import cv2
import numpy as np
import psutil
import requests
import subprocess
import tempfile

import config
import tools.infer.utility as utility

from datetime import datetime
from docx import Document   # python-docx
from flask import Flask, request, jsonify
from tools.infer.predict_system import TextSystem
from urllib.parse import urlparse
from util import process_images, get_image_from_minio

app = Flask(__name__)

config_dict = config.load_config(config_file='local.yml')

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
    example:
    curl -X POST http://192.168.1.109:4101/ocr -H "Content-Type: application/json" -d '{"image_url": "/temp/11.jpg", "source_type": "nas"}'
    :return:
    """
    print("Headers:\n", request.headers)  # 打印请求头
    sys.stdout.flush()

    if request.is_json:
        print("Request is JSON")
    else:
        print("Request is NOT JSON")

    print('request.json', request.json)
    image_url = request.json.get('image_url')
    source_type = request.json.get('source_type', 'minio')
    print('source_type', source_type)
    sys.stdout.flush()

    if not image_url:
        return jsonify({'error': 'Missing image_url parameter'}), 400

    time1 = time.time()

    image_path = None

    # Handling NAS and MinIO sources
    if source_type == 'nas':
        image_url = image_url.replace(config_dict['HIDDEN_PATH'], "")
        print('image_url', image_url)
        sys.stdout.flush()

        if os.path.isabs(image_url):
            if os.path.exists(image_url):
                image_path = image_url
            else:
                return jsonify({'error': 'File not found on server'}), 404
        else:
            response = requests.get(image_url)
            parsed_url = urlparse(image_url)
            print('parsed_url', parsed_url)
            sys.stdout.flush()

            _, file_ext = os.path.splitext(parsed_url.path)
            if not file_ext.startswith('.'):
                file_ext = '.jpg'

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(response.content)
                    image_path = temp_file.name
                    print('image_path', image_path)
            else:
                return jsonify({'error': 'Failed to download image from URL'}), response.status_code
    elif source_type == 'minio':
        print('image_url', image_url)
        image_path = get_image_from_minio(image_url, config_dict)
        if image_path and os.path.exists(image_path):
            print(f"Image exists at {image_path}")
            print(f"File size: {os.path.getsize(image_path)} bytes")
        else:
            return jsonify({'error': 'Image file not found'}), 404

    try:
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 500

        height, width = image.shape[:2]
        print(f'Image dimension: {width}x{height}')
        sys.stdout.flush()

        result, predict_time = process_images(
            text_sys,
            image_dir=image_path,
            process_id=0,
            total_process_num=1,
            page_num=10,
        )

        time2 = time.time()
        print(f'OCR指令执行时间： {time2 - time1}秒')
        sys.stdout.flush()

        parsed_results = parse_ocr_results(result)
        if not parsed_results or 'results' not in parsed_results[0] or not parsed_results[0]['results']:
            return jsonify({
                "width": width,
                "height": height,
                "boxes_num": 0,
                "predict_time": predict_time,
                "results": [],
                "message": "No text detected in the image"
            }), 200

        results = parsed_results[0]['results']
        confidence_threshold = 0.3
        low_confidence = all(float(res_['score']) < confidence_threshold for res_ in results)

        ocr_json = {
            "width": width,
            "height": height,
            "boxes_num": len(results),
            "predict_time": predict_time,
            "results": results
        }

        if low_confidence:
            ocr_json["warning"] = "Low confidence in text detection, possible missed text"

        print('ocr_json', ocr_json)
        sys.stdout.flush()

        return jsonify(ocr_json), 200

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error message: {e.stderr}")
        sys.stdout.flush()
        return jsonify({'error': 'OCR processing failed', 'stderr': e.stderr}), 500

    except Exception as e:
        # If an error occurs, print out system resource usage
        print("Exception occurred:", str(e))
        print("Printing system resource usage:")

        # Print CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")

        # Print memory usage
        memory_info = psutil.virtual_memory()
        print(f"Memory Usage: {memory_info.percent}% used, {memory_info.available / (1024 ** 2):.2f} MB available")

        # Print disk usage
        disk_usage = psutil.disk_usage('/')
        print(f"Disk Usage: {disk_usage.percent}% used, {disk_usage.free / (1024 ** 2):.2f} MB free")

        # Print network statistics
        net_io = psutil.net_io_counters()
        print(f"Network Sent: {net_io.bytes_sent / (1024 ** 2):.2f} MB, Received: {net_io.bytes_recv / (1024 ** 2):.2f} MB")

        sys.stdout.flush()

        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500


@app.route('/current-time', methods=['GET'])
def current_time():
    """
    测试方法：
    http://192.168.1.109:4101/current-time
    """
    now = datetime.now()
    return jsonify({'current_time': now.strftime("%Y-%m-%d %H:%M:%S")}), 200


@app.route('/fetch-images-from-document', methods=['POST'])
def save_docx_images():
    content = request.json
    docx_path = content['file_path']
    output_dir = content['output_dir']

    print('docx_path', docx_path)
    print('output_dir', output_dir)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response = requests.get(docx_path)
    if response.status_code == 200:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(response.content)
            docx_file_path = temp_file.name
            print('docx_file_path', docx_file_path)

        try:
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
                    print('image_path', image_path)
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_blob)
                    images_saved.append(image_path)

            # 删除临时文件
            os.remove(docx_file_path)
            return jsonify({"message": "Images saved successfully", "images": images_saved})
        except Exception as e:
            print(f"Load fail: {e}")
            return jsonify({"message": "Failed to load the docx file"}), 400
    else:
        return jsonify({"message": "Failed to download the docx file"}), 400


if __name__ == '__main__':
    print('Version', 3.1)
    app.run(host='0.0.0.0', port=4101)
