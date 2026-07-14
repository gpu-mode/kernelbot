# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py                         |      107 |       14 |     87% |43-44, 70, 119-126, 258-260, 290-292 |
| src/libkernelbot/background\_submission\_manager.py |      266 |       58 |     78% |39, 41-43, 45, 48, 50, 55, 64-70, 85-86, 89, 99, 111, 172-173, 183, 256-257, 261-271, 288-289, 335-342, 354-356, 365-367, 370-373, 387-392, 419-421, 438-439, 458-459 |
| src/libkernelbot/consts.py                          |       71 |        1 |     99% |        50 |
| src/libkernelbot/db\_types.py                       |       15 |        1 |     93% |         7 |
| src/libkernelbot/hf\_export.py                      |       77 |        4 |     95% |62, 84, 150, 179 |
| src/libkernelbot/kernelguard.py                     |       97 |       44 |     55% |46-48, 52-53, 57-58, 66-71, 75-78, 82-118, 128, 156-159 |
| src/libkernelbot/leaderboard\_db.py                 |      495 |       84 |     83% |66, 101, 412-422, 696, 743-744, 795, 830-831, 842-857, 876-877, 1034-1036, 1050, 1139-1164, 1380-1399, 1579-1603, 1615-1654, 1661-1682, 1689-1696, 1712-1721, 1730-1740, 1748-1758, 1766-1775, 1798 |
| src/libkernelbot/problem\_sync.py                   |      128 |      105 |     18% |72-101, 121-206, 235-302 |
| src/libkernelbot/report.py                          |      269 |        9 |     97% |75, 326, 345, 356, 395, 422, 429-430, 437 |
| src/libkernelbot/submission.py                      |      141 |        7 |     95% |18, 58, 76-81, 88 |
| src/libkernelbot/task.py                            |      113 |        6 |     95% |68, 121, 126-128, 167 |
| src/libkernelbot/utils.py                           |      104 |       11 |     89% |49-50, 64-69, 89-91 |
| **TOTAL**                                           | **1883** |  **344** | **82%** |           |


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