# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/gpu-mode/kernelbot/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/libkernelbot/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| src/libkernelbot/backend.py                         |       99 |       13 |     87% |43-44, 94-101, 233-235, 265-267 |
| src/libkernelbot/background\_submission\_manager.py |      169 |       32 |     81% |37, 39-41, 43, 46, 48, 177-178, 204-207, 225-230, 248-250, 267-288 |
| src/libkernelbot/consts.py                          |       71 |        1 |     99% |        50 |
| src/libkernelbot/db\_types.py                       |       15 |        1 |     93% |         7 |
| src/libkernelbot/hf\_export.py                      |       77 |        4 |     95% |62, 84, 150, 179 |
| src/libkernelbot/kernelguard.py                     |       97 |       47 |     52% |43-48, 52-53, 57-58, 66-71, 75-78, 82-118, 128, 156-159 |
| src/libkernelbot/leaderboard\_db.py                 |      489 |       89 |     82% |66, 101, 381-405, 412-422, 669, 716-717, 768, 803-804, 815-830, 849-850, 959-961, 975, 1064-1089, 1286-1318, 1445-1469, 1481-1520, 1527-1548, 1555-1562, 1578-1587, 1596-1606, 1614-1624, 1632-1641, 1664 |
| src/libkernelbot/problem\_sync.py                   |      128 |      105 |     18% |72-101, 121-206, 235-302 |
| src/libkernelbot/report.py                          |      269 |        9 |     97% |75, 326, 345, 356, 395, 422, 429-430, 437 |
| src/libkernelbot/submission.py                      |      140 |       10 |     93% |18, 54, 72-77, 81-84 |
| src/libkernelbot/task.py                            |      113 |        6 |     95% |68, 121, 126-128, 167 |
| src/libkernelbot/utils.py                           |      104 |       11 |     89% |49-50, 64-69, 89-91 |
| **TOTAL**                                           | **1771** |  **328** | **81%** |           |


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