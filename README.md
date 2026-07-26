# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/application\_validation.py         |      141 |       50 |     65% |26, 42-45, 51-61, 64-71, 81, 89, 108-119, 122-124, 147, 150, 161, 169, 175, 203-223, 244, 249, 317-325 |
| src/libkernelbot/backend.py                         |      107 |       14 |     87% |43-44, 70, 119-126, 258-260, 290-292 |
| src/libkernelbot/background\_submission\_manager.py |      266 |       58 |     78% |39, 41-43, 45, 48, 50, 55, 64-70, 85-86, 89, 99, 111, 172-173, 183, 256-257, 261-271, 288-289, 335-342, 354-356, 365-367, 370-373, 387-392, 419-421, 438-439, 458-459 |
| src/libkernelbot/consts.py                          |       71 |        1 |     99% |        50 |
| src/libkernelbot/db\_types.py                       |       15 |        1 |     93% |         7 |
| src/libkernelbot/hf\_export.py                      |       77 |        4 |     95% |62, 84, 150, 179 |
| src/libkernelbot/kernelguard.py                     |       97 |       44 |     55% |46-48, 52-53, 57-58, 66-71, 75-78, 82-118, 128, 156-159 |
| src/libkernelbot/leaderboard\_db.py                 |      524 |       88 |     83% |66, 101, 412-422, 696, 743-744, 795, 830-831, 842-857, 876-877, 926, 958, 970, 998, 1188-1190, 1204, 1293-1318, 1534-1553, 1733-1757, 1769-1808, 1815-1836, 1843-1850, 1866-1875, 1884-1894, 1902-1912, 1920-1929, 1952 |
| src/libkernelbot/problem\_sync.py                   |      128 |      105 |     18% |72-101, 121-206, 235-302 |
| src/libkernelbot/report.py                          |      269 |        9 |     97% |75, 326, 345, 356, 395, 422, 429-430, 437 |
| src/libkernelbot/submission.py                      |      141 |        7 |     95% |18, 58, 76-81, 88 |
| src/libkernelbot/task.py                            |      131 |        8 |     94% |37, 79, 134, 139-141, 184, 247 |
| src/libkernelbot/utils.py                           |      104 |       11 |     89% |49-50, 64-69, 89-91 |
| src/libkernelbot/validation\_runtime.py             |       62 |        8 |     87% |37, 49, 54, 70, 72, 74, 111-112 |
| **TOTAL**                                           | **2133** |  **408** | **81%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/gpu-mode/kernelbot/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/gpu-mode/kernelbot/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgpu-mode%2Fkernelbot%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.