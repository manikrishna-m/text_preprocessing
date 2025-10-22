import io
import cv2
import fitz
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes


class OCRPreprocessorBytes:
    def __init__(self, dpi=300):
        self.dpi = dpi

    # ---------------------------
    # 1. Detect if PDF is digital-born
    # ---------------------------
    def is_digital_pdf(self, pdf_bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_area = 0
        text_area = 0
        for page in doc:
            total_area += abs(page.rect)
            blocks = page.get_text("blocks")
            for b in blocks:
                rect = fitz.Rect(b[:4])
                text_area += abs(rect)
        coverage = text_area / total_area if total_area > 0 else 0
        return coverage >= 0.01  # True if digital-born

    # ---------------------------
    # 2. Convert PDF bytes to high-res images
    # ---------------------------
    def pdf_bytes_to_images(self, pdf_bytes):
        pil_images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
        pil_images = [img.convert('RGB') for img in pil_images]
        return pil_images

    # ---------------------------
    # 3. Convert PIL to OpenCV
    # ---------------------------
    def pil_to_cv(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---------------------------
    # 4. Grayscale conversion
    # ---------------------------
    def to_grayscale(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # ---------------------------
    # 5. Denoising
    # ---------------------------
    def denoise(self, gray_img, method='median', **kwargs):
        if method == 'median':
            k = kwargs.get('ksize', 3)
            return cv2.medianBlur(gray_img, k)
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigmaColor = kwargs.get('sigmaColor', 75)
            sigmaSpace = kwargs.get('sigmaSpace', 75)
            return cv2.bilateralFilter(gray_img, d, sigmaColor, sigmaSpace)
        elif method == 'nlmeans':
            h = kwargs.get('h', 10)
            return cv2.fastNlMeansDenoising(gray_img, None, h, 7, 21)
        else:
            raise ValueError("Unsupported denoise method")

    # ---------------------------
    # 6. Binarization
    # ---------------------------
    def binarize(self, gray_img, method='otsu', block_size=31, C=10):
        if method == 'otsu':
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C)
        elif method == 'fixed':
            _, binary = cv2.threshold(gray_img, C, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError("Unsupported binarization method")
        return binary

    # ---------------------------
    # 7. Contrast enhancement
    # ---------------------------
    def enhance_contrast(self, gray_img, method='clahe'):
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(gray_img)
        elif method == 'hist_eq':
            return cv2.equalizeHist(gray_img)
        else:
            return gray_img

    # ---------------------------
    # 8. Skew correction
    # ---------------------------
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

    # ---------------------------
    # 9. Full preprocessing pipeline for a single image
    # ---------------------------
    def preprocess_image(self, img, denoise_method='median', binarize_method='otsu', enhance_method='clahe', deskew_flag=True):
        gray = self.to_grayscale(img)
        denoised = self.denoise(gray, method=denoise_method)
        enhanced = self.enhance_contrast(denoised, method=enhance_method)
        binary = self.binarize(enhanced, method=binarize_method)
        if deskew_flag:
            final_img = self.deskew(binary, enhanced)
        else:
            final_img = enhanced
        return final_img, binary

    # ---------------------------
    # 10. Main processing entry point
    # ---------------------------
    def process_bytes(self, file_bytes, file_type='pdf'):
        """
        file_type: 'pdf', 'image' (png, jpeg, jpg, tiff)
        """
        preprocessed_images = []
        digital_text = None

        if file_type == 'pdf':
            if self.is_digital_pdf(file_bytes):
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                digital_text = ""
                for page in doc:
                    digital_text += page.get_text()
                return {'text': digital_text, 'images': None}
            else:
                pil_images = self.pdf_bytes_to_images(file_bytes)
                for pil_img in pil_images:
                    img = self.pil_to_cv(pil_img)
                    final_img, binary = self.preprocess_image(img)
                    preprocessed_images.append(final_img)
        elif file_type == 'image':
            img_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            final_img, binary = self.preprocess_image(img)
            preprocessed_images.append(final_img)
        else:
            raise ValueError("Unsupported file_type")

        return {'text': digital_text, 'images': preprocessed_images}
