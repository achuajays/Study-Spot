from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import os


class ImageOCR:
    def __init__(self):
        self.ocr = PaddleOCR()

    def extract_text(self, image_path):
        try:
            result = self.ocr.ocr(image_path)
            extracted_text = ' '.join([word[1][0] for line in result for word in line])
            try:
                os.remove(image_path)
            except Exception as e:
                print(e)
                
            return extracted_text
        except Exception as e:
            print("Error:", e)
            return None