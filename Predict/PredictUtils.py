# -*- coding: utf-8 -*-
import os


def updataLabel(file):
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if '>' in line:
                line = line.replace('\n','|0|test\n')
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)