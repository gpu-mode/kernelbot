# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py                         |       82 |        9 |     89% |38-39, 62, 203-205, 235-237 |
| src/libkernelbot/background\_submission\_manager.py |      160 |       31 |     81% |36, 38-40, 42, 45, 47, 176-177, 203-206, 224-229, 246-271 |
| src/libkernelbot/consts.py                          |       65 |        1 |     98% |        48 |
| src/libkernelbot/db\_types.py                       |       14 |        1 |     93% |         7 |
| src/libkernelbot/leaderboard\_db.py                 |      333 |       50 |     85% |65, 99, 373-383, 396-414, 719-721, 735, 824-849, 1090-1114, 1126-1165, 1172-1193, 1200-1207, 1223-1232 |
| src/libkernelbot/problem\_sync.py                   |      128 |      105 |     18% |72-101, 121-206, 234-300 |
| src/libkernelbot/report.py                          |      269 |        9 |     97% |75, 326, 345, 356, 395, 422, 429-430, 437 |
| src/libkernelbot/submission.py                      |      121 |        1 |     99% |        18 |
| src/libkernelbot/task.py                            |      112 |        6 |     95% |68, 121, 126-128, 165 |
| src/libkernelbot/utils.py                           |      104 |       11 |     89% |49-50, 64-69, 89-91 |
| **TOTAL**                                           | **1388** |  **224** | **84%** |           |


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