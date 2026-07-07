"""Image loading helpers with camera EXIF orientation handling."""
import cv2


def exif_orientation_to_cv2_code(orientation):
    if orientation == 3:
        return cv2.ROTATE_180
    if orientation == 6:
        return cv2.ROTATE_90_CLOCKWISE
    if orientation == 8:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _read_exif_orientation(path):
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif = img.getexif()
            return exif.get(274) if exif else None
    except Exception:
        return None


def force_landscape(image):
    """Rotate portrait images to landscape before detection."""
    if image is None:
        return None
    h, w = image.shape[:2]
    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def load_bgr_with_orientation(path, force_landscape_mode=True):
    image = cv2.imread(path)
    if image is None:
        return None
    rotate_code = exif_orientation_to_cv2_code(_read_exif_orientation(path))
    if rotate_code is not None:
        image = cv2.rotate(image, rotate_code)
    if force_landscape_mode:
        image = force_landscape(image)
    return image
