import io
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageEnhance

class PDFPreprocessor:
    def __init__(self, dpi=300, contrast_factor=1.5):
        self.dpi = dpi
        self.contrast_factor = contrast_factor

    # --- Step 1: Convert PDF bytes -> Images ---
    def _pdf_to_images(self, pdf_bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            zoom = self.dpi / 72.0  # scale factor
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
        return images

    # --- Step 2: Enhance contrast ---
    def _adjust_contrast(self, image):
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(self.contrast_factor)
        return np.array(enhanced)

    # --- Step 3: Deskew ---
    def _deskew(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # --- Step 4: Remove background ---
    def _remove_background(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cleaned = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    # --- Step 5: Orientation correction (simple heuristic) ---
    def _correct_rotation(self, image):
        # Simple text-orientation correction (if needed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        if lines is not None:
            angles = [line[0][1] for line in lines]
            avg_angle = np.mean(angles)
            angle_deg = np.degrees(avg_angle - np.pi/2)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        return image

    # --- Step 6: Convert back to PDF bytes ---
    def _images_to_pdf(self, images):
        pil_images = [Image.fromarray(img).convert("RGB") for img in images]
        output = io.BytesIO()
        pil_images[0].save(output, format="PDF", save_all=True, append_images=pil_images[1:])
        return output.getvalue()

    # --- MAIN PIPELINE ---
    def preprocess_pdf(self, pdf_bytes):
        pages = self._pdf_to_images(pdf_bytes)
        processed_pages = []
        for img in pages:
            img = self._adjust_contrast(img)
            img = self._deskew(img)
            img = self._remove_background(img)
            img = self._correct_rotation(img)
            processed_pages.append(img)
        cleaned_pdf = self._images_to_pdf(processed_pages)
        return cleaned_pdf, processed_pages
