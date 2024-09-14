# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import gc
import json
import logging
import numpy as np
import time

import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls

from PIL import Image
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("No valid image provided")
            return None, None, time_dict

        # Optional image downscaling to limit memory usage
        max_size = 1024  # Define maximum allowed image dimension
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale_factor = max_size / max(height, width)
            img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
            logger.debug(f"Resized image to {img.shape}")

        start = time.time()
        ori_im = img.copy()

        try:
            dt_boxes, elapse = self.text_detector(img)
            time_dict['det'] = elapse

            if dt_boxes is None:
                logger.debug(f"No dt_boxes found, elapsed: {elapse}")
                time_dict['all'] = time.time() - start
                return None, None, time_dict
            else:
                logger.debug(f"dt_boxes num: {len(dt_boxes)}, elapsed: {elapse}")
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            gc.collect()
            return None, None, time_dict

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)

        # Cropping detected text regions
        for bno, tmp_box in enumerate(dt_boxes):
            try:
                if self.args.det_box_type == "quad":
                    img_crop = get_rotate_crop_image(ori_im, tmp_box)
                else:
                    img_crop = get_minarea_rect_crop(ori_im, tmp_box)

                img_crop_list.append(img_crop)
                logger.debug(f"Cropped image {bno} successfully")
            except Exception as e:
                logger.error(f"Error cropping image {bno}: {e}")
                gc.collect()
                continue

        logger.debug(f"Cropped {len(img_crop_list)} images, preparing for classification")

        # Classify the cropped images (optional step)
        if self.use_angle_cls and cls:
            try:
                img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
                time_dict['cls'] = elapse
                logger.debug(f"Classified {len(img_crop_list)} images, elapsed: {elapse}")
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                gc.collect()
                return None, None, time_dict

        # Recognize text in cropped images
        try:
            rec_res, elapse = self.text_recognizer(img_crop_list)
            time_dict['rec'] = elapse
            logger.debug(f"rec_res num: {len(rec_res)}, elapsed: {elapse}")
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            gc.collect()
            return None, None, time_dict

        # Optionally save cropped images
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)

        # Filter results based on score
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict['all'] = end - start

        # Perform garbage collection to release resources
        gc.collect()

        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = False
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    for idx, image_file in enumerate(image_file_list):

        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            if len(imgs) > 1:
                logger.debug(
                    str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                    % (image_file, elapse))
            else:
                logger.debug(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file,
                                                                elapse))
            for text, score in rec_res:
                logger.debug("{}, {:.3f}".format(text, score))

            res = [{
                "transcription": rec_res[i][0],
                "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
            } for i in range(len(dt_boxes))]
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + json.dumps(
                        res, ensure_ascii=False) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                    res, ensure_ascii=False) + "\n"
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path)
                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace('.pdf', '_' + str(index) + '.png')
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                    draw_img[:, :, ::-1])
                logger.debug("The visualized image saved in {}".format(
                    os.path.join(draw_img_save_dir, os.path.basename(
                        save_file))))

    logger.info("The predict total time is {}".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    with open(
            os.path.join(draw_img_save_dir, "system_results.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)


if __name__ == "__main__":
    # --image_dir="resources/imgs/11.jpg" --det_model_dir="inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True
    # --image_dir="resources/imgs/11.jpg" --det_model_dir="inference/ch_PP-OCRv3_det_infer/" --rec_model_dir="inference/ch_PP-OCRv3_rec_infer/" --cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # 改变当前工作目录
    os.chdir(project_root)

    time1 = time.time()

    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num

        print('predict_system.py', 'total_process_num', total_process_num)
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)

        print('predict_system.py', 'p_list', p_list)
        for p in p_list:
            p.wait()
    else:
        print('args', args)
        main(args)

    time2 = time.time()
    print(f'OCR指令执行时间： {time2 - time1}秒')

    # input("Press Enter to exit...")
