import os
import subprocess
import time
import cv2
import numpy as np

from tools.infer import utility
from tools.infer.predict_system import TextSystem
from util import process_images, parse_ocr_results

# 将工作目录切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 解析命令行参数
args = utility.parse_args()

args.det_model_dir = 'inference/ch_PP-OCRv3_det_infer/'
args.rec_model_dir = 'inference/ch_PP-OCRv3_rec_infer/'
args.cls_model_dir = 'inference/ch_ppocr_mobile_v2.0_cls_infer/'
args.use_angle_cls = True
args.use_space_char = True

# 使用解析后的参数
text_sys = TextSystem(args)


def imread_unicode(image_path_):
    """
    使用 cv2.imdecode 读取包含中文路径的图片
    :param image_path_: 图片路径
    :return: OpenCV 图像对象
    """
    with open(image_path_, "rb") as f:
        image_data = f.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def read(image_path_):
    temp_image_path = None
    try:
        # 使用自定义的 imread_unicode 读取图片
        image = imread_unicode(image_path_)
        if image is None:
            print("无法读取图片")
            return

        # 将图像保存为临时文件，确保图像被正确读取
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, image)
        print(f"图片已保存为 {temp_image_path} 以供调试使用")

        height, width = image.shape[:2]
        print(f'Image dimension: {width}x{height}')

        time1 = time.time()
        result, predict_time = process_images(
            text_sys,
            image_dir=temp_image_path,  # 使用保存的临时文件进行OCR处理
            process_id=0,
            total_process_num=1,
            page_num=10,
        )

        time2 = time.time()
        print(f'OCR指令执行时间： {time2 - time1}秒')

        if not result:
            print("OCR未返回任何结果")
            return

        parsed_results = parse_ocr_results(result)

        if not parsed_results:
            print("OCR解析结果为空")
            return

        results = parsed_results[0]['results']
        print('results', results)

        ocr_json = {
            "width": width,
            "height": height,
            "boxes_num": len(results),
            "predict_time": predict_time,
            "results": results
        }
        print('ocr_json', ocr_json)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error message: {e.stderr}")

    finally:
        # 删除临时文件
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"临时文件 {temp_image_path} 已删除")


if __name__ == "__main__":
    image_path = r"E:/个人文档/七小鹿/2.jpg"

    if os.path.exists(image_path):
        print("文件存在")
    else:
        print("文件不存在，请检查路径")

    read(image_path)
