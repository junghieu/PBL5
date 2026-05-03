# -*- coding: utf-8 -*-
import re

filepath = r"D:\PBL5\PBL5\server_pc\pipeline\ocr.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

pattern1 = r"(config\['predictor'\]\['beamsearch'\].*?\n)"
replace1 = r"\g<1>    config['dataset']['max_seq_length'] = 128\n    if 'length_penalty' not in config['predictor']: config['predictor']['length_penalty'] = 1.2\n"
content = re.sub(pattern1, replace1, content, count=1)

pattern2 = r"(pred_text\s*=\s*ocr\.predict\(pil_img\))"
replace2 = r"\g<1>\n        pred_text = process_ocr_text(str(pred_text))"
content = re.sub(pattern2, replace2, content, count=1)

pattern3 = r"(pred\s*=\s*ocr\.predict\(Image\.fromarray\(cv2\.cvtColor\(img,\s*cv2\.COLOR_BGR2RGB\)\)\))(\.strip\(\))"
replace3 = r"\g<1>\n        pred = process_ocr_text(str(pred))\g<2>"
content = re.sub(pattern3, replace3, content, count=1)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print('Patch applied successfully')
