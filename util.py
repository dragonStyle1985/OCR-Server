import os
import json
import numpy as np
import cv2
import time

from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read, get_image_file_list
logger = get_logger()


def process_images(
    text_sys,
    image_dir,
    process_id,
    total_process_num,
    page_num=0,
):
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list[process_id::total_process_num]
    is_visualize = False
    # os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    # 处理图片
    total_time = time.time()
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)  # 假设有这个函数
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)  # 路径不支持中文
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]

        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            logger.debug(f"{idx}_{index if len(imgs) > 1 else ''}  Predict time of {image_file}: {elapse:.3f}s")
            score_list = []
            for text, score in rec_res:
                score_list.append(score)
                logger.debug(f"{text}, {score:.3f}")
            res = [{
                "transcription": rec_res[i][0],
                "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                "score": f"{score_list[i]:.3f}"
            } for i in range(len(dt_boxes))]
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                    res, ensure_ascii=False) + "\n"
            save_results.append(save_pred)

    predict_time = time.time() - total_time
    logger.info(f"The predict total time is {predict_time}")

    return save_results, predict_time