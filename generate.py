#!/usr/bin/env python

import logging as log
import datetime

from pathlib import Path
from PIL import Image
from video_kandinsky3 import get_T2V_pipeline

def generate_video() -> None:
    log.basicConfig(level = log.DEBUG, format = '%(asctime)s   %(levelname)s   %(message)s')
    log.info('Start')

    cache_dir = Path('./cache')
    cache_dir.mkdir(exist_ok = True)

    result_dir = Path('./result')
    result_dir.mkdir(exist_ok = True)

    log.info('Loading pipeline...')
    t2v_pipe = get_T2V_pipeline('cuda', fp16 = True, cache_dir = str(cache_dir))

    log.info('Generating video...')
    video = t2v_pipe(
        'a red car is drifting on the mountain road, close view, fast movement',
        width = 640,
        height = 384,
        fps = 'low' # low, medium, high
    )

    result_file = Path(result_dir, f'video-{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")}.gif')
    while result_file.exists():
        result_file = Path(result_dir, f'video-{datetime.datetime.now().strftime("$Y-%m-%dT%H-%M-%S-%f")}.gif')

    video[0].save(str(result_file), save_all = True, append_images = video[1:], optimize = False, duration = 7, loop = 0)
    log.info(f'Result has been saved in file {result_file}')

    log.info('Finish')

if __name__ == '__main__':
    generate_video()
