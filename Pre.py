import io
import fitz
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict
from aiutils.file_utils import FileType, get_file_type_from_bytes


class Preprocessor:
    """
    OCR preprocessing:
    - Accepts PDF or image bytes
    - Converts pages/images to grayscale
    - Denoises, enhances contrast (CLAHE)
    - Applies adaptive thresholding
    - Optional deskew
    - Returns preprocessed PDF bytes or images
    """

    def __init__(self, dpi: int = 300, deskew_flag: bool = True):
        self.dpi = dpi
        self.deskew_flag = deskew_flag

    # ---------------------------
    # Grayscale conversion
    # ---------------------------
    def convert_to_grayscale(self, rgb: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        return np.ascontiguousarray(ycc[..., 0], dtype=np.uint8)

    # ---------------------------
    # Denoising
    # ---------------------------
    def denoising(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # ---------------------------
    # Contrast Enhancement (CLAHE)
    # ---------------------------
    def enhance_contrast(self, gray_img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_img)

    # ---------------------------
    # Adaptive Binarization
    # ---------------------------
    def binarize(self, gray_img: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )

    # ---------------------------
    # Deskewing
    # ---------------------------
    def deskew(self, binary_img: np.ndarray, orig_img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(binary_img < 255))
        if len(coords) == 0:
            return orig_img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        (h, w) = orig_img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(orig_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # ---------------------------
    # Preprocess PDF bytes
    # ---------------------------
    def bytes_preprocessing_pdf(self, doc_bytes: bytes) -> List[np.ndarray]:
        pages: List[np.ndarray] = []

        with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]

                image = self.convert_to_grayscale(arr)
                image = self.denoising(image)
                image = self.enhance_contrast(image)
                binary = self.binarize(image)

                if self.deskew_flag:
                    image = self.deskew(binary, image)

                pages.append(image)
        return pages

    # ---------------------------
    # Preprocess Image bytes
    # ---------------------------
    def bytes_preprocessing_image(self, img_bytes: bytes) -> List[np.ndarray]:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")

        image = self.convert_to_grayscale(img)
        image = self.denoising(image)
        image = self.enhance_contrast(image)
        binary = self.binarize(image)

        if self.deskew_flag:
            image = self.deskew(binary, image)

        return [image]

    # ---------------------------
    # Convert pages to PDF bytes
    # ---------------------------
    def get_bytes_postprocessing(self, pages: List[np.ndarray]) -> bytes:
        if not pages:
            return b""

        pil_pages = []
        for p in pages:
            img = Image.fromarray(p)
            if img.mode != "L":
                img = img.convert("L")
            pil_pages.append(img)

        buf = io.BytesIO()
        pil_pages[0].save(
            buf,
            format="PDF",
            save_all=True,
            append_images=pil_pages[1:],
            resolution=float(self.dpi),
        )
        return buf.getvalue()

    # ---------------------------
    # Main Entry
    # ---------------------------
    def __call__(self, doc_bytes: bytes) -> bytes:
        file_type = get_file_type_from_bytes(doc_bytes)
        pages: List[np.ndarray] = []

        if file_type == FileType.PDF:
            pages = self.bytes_preprocessing_pdf(doc_bytes)
        elif file_type in {FileType.JPEG, FileType.TIFF, FileType.PNG}:
            pages = self.bytes_preprocessing_image(doc_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return self.get_bytes_postprocessing(pages)


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    preprocessor = Preprocessor(dpi=300)

    # Example: PDF bytes
    with open("example.pdf", "rb") as f:
        pdf_bytes = f.read()
    preprocessed_pdf_bytes = preprocessor(pdf_bytes)
    with open("preprocessed_output.pdf", "wb") as f:
        f.write(preprocessed_pdf_bytes)

    # Example: Image bytes
    with open("scan.jpg", "rb") as f:
        img_bytes = f.read()
    preprocessed_img_bytes = preprocessor(img_bytes)
    with open("preprocessed_image.pdf", "wb") as f:
        f.write(preprocessed_img_bytes)
