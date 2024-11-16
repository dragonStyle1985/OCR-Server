import os
import cv2
import numpy as np

from local.ocr import imread_unicode
from tools.infer import utility
from tools.infer.predict_system import TextSystem
from util import process_images, parse_ocr_results

# 将工作目录切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 初始化OCR模型
args = utility.parse_args()
args.det_model_dir = 'inference/ch_PP-OCRv3_det_infer/'
args.rec_model_dir = 'inference/ch_PP-OCRv3_rec_infer/'
args.cls_model_dir = 'inference/ch_ppocr_mobile_v2.0_cls_infer/'
args.use_angle_cls = True
args.use_space_char = True
text_sys = TextSystem(args)


def crop_and_ocr(image_path, crop_height=1000):
    """
    裁剪图片并逐块进行OCR识别
    :param image_path: 图片路径
    :param crop_height: 每块裁剪的高度（默认1000像素）
    """
    image = imread_unicode(image_path)
    if image is None:
        print("无法读取图片")
        return

    height, width = image.shape[:2]
    print(f"图片原始尺寸: 宽 {width}, 高 {height}")

    results = []
    for start_y in range(0, height, crop_height):
        end_y = min(start_y + crop_height, height)
        cropped_image = image[start_y:end_y, :]

        # 保存裁剪块用于调试（可选）
        temp_image_path = f"temp_crop_{start_y}.jpg"
        cv2.imwrite(temp_image_path, cropped_image)
        print(f"已裁剪图片块: {temp_image_path}, 高度范围: {start_y}-{end_y}")

        # OCR识别
        result, predict_time = process_images(
            text_sys,
            image_dir=temp_image_path,  # 使用裁剪后的图片
            process_id=0,
            total_process_num=1,
            page_num=10,
        )
        parsed_results = parse_ocr_results(result)

        if parsed_results:
            results.extend(parsed_results[0]['results'])

        # 删除临时文件
        os.remove(temp_image_path)

    return results


if __name__ == "__main__":

    # image_path = r"E:\个人文档\房子\装修\方林\合同\装饰材料代理购买协议——关于装修装饰工程合同满额减模式的补充协议.jpg"
    image_path = r"E:\个人文档\房子\装修\方林\合同\装饰装修工程施工合同1.jpg"

    if os.path.exists(image_path):
        print("文件存在，开始裁剪和OCR识别...")
        ocr_results = crop_and_ocr(image_path)

        # 打印最终识别结果
        if ocr_results:
            print("OCR识别结果：")
            for res in ocr_results:
                print(res)
        else:
            print("未识别到任何文本")
    else:
        print("文件不存在，请检查路径")
