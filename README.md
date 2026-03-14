# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py                         |       82 |        9 |     89% |38-39, 62, 203-205, 235-237 |
| src/libkernelbot/background\_submission\_manager.py |      160 |       31 |     81% |36, 38-40, 42, 45, 47, 176-177, 203-206, 224-229, 246-271 |
| src/libkernelbot/consts.py                          |       67 |        1 |     99% |        50 |
| src/libkernelbot/db\_types.py                       |       14 |        1 |     93% |         7 |
| src/libkernelbot/hf\_export.py                      |       77 |        4 |     95% |62, 84, 150, 179 |
| src/libkernelbot/leaderboard\_db.py                 |      404 |       67 |     83% |65, 100, 374-384, 397-415, 631, 678-679, 730, 765-766, 777-792, 811-812, 921-923, 937, 1026-1051, 1292-1316, 1328-1367, 1374-1395, 1402-1409, 1425-1434 |
| src/libkernelbot/problem\_sync.py                   |      128 |      105 |     18% |72-101, 121-206, 235-302 |
| src/libkernelbot/report.py                          |      269 |        9 |     97% |75, 326, 345, 356, 395, 422, 429-430, 437 |
| src/libkernelbot/submission.py                      |      130 |        5 |     96% | 18, 67-72 |
| src/libkernelbot/task.py                            |      113 |        6 |     95% |68, 121, 126-128, 167 |
| src/libkernelbot/utils.py                           |      104 |       11 |     89% |49-50, 64-69, 89-91 |
| **TOTAL**                                           | **1548** |  **249** | **84%** |           |


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