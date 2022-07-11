"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 10:50:20
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re

FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './docs/{}/README.md'
MAX_SPACE = 20


def read_comments(file) -> list[str]:
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = ''.join(lines)
    args = re.findall('@property[^@]*', lines)

    results = []
    for arg in args:
        name = re.findall('(def )(.+)(\()', arg)[0][1]
        dtype = re.findall('(-> )(.*)(:)', arg)[0][1]
        argtype = re.findall('(argtype=)(.*)(\))', arg)[0][1]
        default = re.findall('(, )(.*)(, arg)', arg)[0][1]
        comments = re.findall('(""")([\S\s]+)(""")', arg)[0][1]
        comments = comments.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            comments = comments.replace('  ', ' ')

        comments = re.findall('( *)(.*)( *)', comments)[0][1]

        if comments.endswith('. '):
            comments = comments[:-1]

        s = '- `--{}`, type=`{}`, argtype=`{}`.\n  {}\n  The default value is `{}`.'.format(
            name, dtype, argtype, comments, default)
        results.append(s + '\n')
        print(s)

    return results


def update(md_file, files: list[str], titles: list[str]):

    new_lines = []
    for f, title in zip(files, titles):
        new_lines += ['\n### {}\n\n'.format(title)]
        c = read_comments(f)
        c.sort()
        new_lines += c

    with open(md_file, 'r') as f:
        lines = f.readlines()
    lines = ''.join(lines)

    try:
        pattern = re.findall(
            '([\s\S]*)({})([\s\S]*)({})([\s\S]*)'.format(FLAG, FLAG), lines)[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])

    except:
        flag_line = '{}\n'.format(FLAG)
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+') as f:
        f.writelines(all_lines)


if __name__ == '__main__':
    for model in ['Vertical']:
        files = ['./codes/args/__args.py',
                 './{}/__args.py'.format(model)]
        titles = ['Basic args',
                  '{} args'.format(model)]
        update(TARGET_FILE.format(model), files, titles)
