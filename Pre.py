import io
import math
import fitz
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict
from aiutils.file_utils import FileType, get_file_type_from_bytes


class Preprocessor:
    """
    PDF -> per-page RGB render with DPI chosen by most frequent font size (mode),
    convert to grayscale (luminance Y), apply CLAHE (local contrast), then save to PDF.
    """

    def __init__(self, dpi = 300):
        self.dpi = dpi  

    def convert_to_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        return np.ascontiguousarray(ycc[..., 0], dtype=np.uint8)
    
    def denoising(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize= 7, searchWindowSize=21)
    
    def binarize(self, gray_img, block_size=31, C=10) -> np.ndarray:
        return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C)
    
    def deskew(self, binary_img, orig_img):
        coords = np.column_stack(np.where(binary_img < 255))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        (h, w) = orig_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(orig_img, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def bytes_preprocessing(self, doc_bytes: bytes) -> List[np.ndarray]:
        pages: List[np.ndarray] = []

        with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if arr.shape[2] == 4: arr = arr[:, :, :3]
                image = self.convert_to_grayscale(arr)
                image = self.denoising(image)
                image = self.binarize(image)

                pages.append(image)      

        return pages

    def get_bytes_postprocessing(self, pages: List[np.ndarray]) -> bytes:
        if not pages:
            return b""

        pil_pages = []
        for p in pages:
            img = Image.fromarray(p)  # 2D uint8 -> 'L'
            if img.mode != "L":
                img = img.convert("L")
            pil_pages.append(img)

        buf = io.BytesIO()
        meta_dpi = float(max(self.dpi))
        pil_pages[0].save(
            buf,
            format="PDF",
            save_all=True,
            append_images=pil_pages[1:],
            resolution=meta_dpi,  # Pillow expects a single numeric value
        )
        return buf.getvalue()

    def __call__(self, doc_bytes: bytes) -> bytes:
        file_type = get_file_type_from_bytes(doc_bytes)
        if file_type == FileType.PDF:
            pages = self.bytes_preprocessing(doc_bytes)
        elif file_type in {FileType.JPEG, FileType.TIFF, FileType.PNG}:
            pass


        return self.get_bytes_postprocessing(pages)
