# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('D:/PBL5/PBL5/server_pc')
from pipeline.post_processing import process_ocr_text

tests = [
     '( text )', 
     '1 - 1963', 
     '1B )', 
     'C )', 
     '1915 today', 
     'Dấu hiệu lặp c c c c'
]
for t in tests:
     print(t, '==>', process_ocr_text(t))
