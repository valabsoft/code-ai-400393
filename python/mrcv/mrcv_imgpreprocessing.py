import cv2
import numpy as np
from enum import Enum
import logging
import os  # Added for file existence check

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Enum definitions (unchanged)
class METOD_IMAGE_PERPROCESSIN(Enum):
    NONE = 0
    CONVERTING_BGR_TO_GRAY = 1
    BRIGHTNESS_LEVEL_UP = 2
    BRIGHTNESS_LEVEL_DOWN = 3
    BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST = 4
    BALANCE_CONTRAST_02_YCBCR_CLAHE = 5
    BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING = 6
    BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION = 7
    BALANCE_CONTRAST_05_HSV_EQUALIZEHIST = 8
    BALANCE_CONTRAST_06_HSV_CLAHE = 9
    BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING = 10
    BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION = 11
    BALANCE_CONTRAST_09_LAB_EQUALIZEHIST = 12
    BALANCE_CONTRAST_10_LAB_CLAHE = 13
    BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING = 14
    BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION = 15
    BALANCE_CONTRAST_13_RGB_EQUALIZEHIST = 16
    BALANCE_CONTRAST_14_RGB_CLAHE = 17
    BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING = 18
    BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION = 19
    SHARPENING_01 = 20
    SHARPENING_02 = 21
    NOISE_FILTERING_01_MEDIAN_FILTER = 22
    NOISE_FILTERING_02_AVARAGE_FILTER = 23
    CORRECTION_GEOMETRIC_DEFORMATION = 24


class METOD_INCREASE_IMAGE_CONTRAST(Enum):
    EQUALIZE_HIST = 0
    CLAHE = 1
    CONTRAST_BALANCING = 2
    CONTRAST_EXTENSION = 3


class COLOR_MODEL(Enum):
    CM_HSV = 0
    CM_LAB = 1
    CM_YCBCR = 2
    CM_RGB = 3


class LOGTYPE(Enum):
    DEBUG = 0
    ERROR = 1


class CameraParameters:
    def __init__(self):
        self.M1 = None
        self.D1 = None
        self.R = None
        self.T = None
        self.R1 = None
        self.P1 = None
        self.imageSize = None
        self.rms = None
        self.avgErr = None
        self.map11 = None
        self.map12 = None


class MRCV:
    @staticmethod
    def getErrorImage(textError: str) -> np.ndarray:
        errorImage = np.zeros((600, 960, 3), dtype=np.uint8)
        cv2.putText(
            errorImage,
            textError,
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (162, 20, 47),  # BGR color
            3,
            cv2.LINE_8,
            False
        )
        return errorImage

    @staticmethod
    def writeLog(message: str, log_type: LOGTYPE = LOGTYPE.DEBUG):
        if log_type == LOGTYPE.ERROR:
            logger.error(message)
        else:
            logger.debug(message)

    @staticmethod
    def changeImageBrightness(imageInput: np.ndarray, gamma: float) -> tuple[int, np.ndarray]:
        try:
            if imageInput.size == 0:
                return 1, MRCV.getErrorImage("changeImageBrightness:: Image is Empty")

            # Gamma correction
            imageDouble = imageInput.astype(np.float64) / 255.0
            imageDouble = np.power(imageDouble, gamma)
            imageDouble = imageDouble * 255.0
            imageOutput = imageDouble.astype(np.uint8)
            return 0, imageOutput
        except Exception as e:
            MRCV.writeLog(f"changeImageBrightness:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1, MRCV.getErrorImage("changeImageBrightness:: Unhandled Exception")

    @staticmethod
    def sharpeningImage01(imageInput: np.ndarray, gainFactorHighFrequencyComponent: float) -> tuple[int, np.ndarray]:
        try:
            if imageInput.size == 0:
                return 1, MRCV.getErrorImage("sharpeningImage01:: Image is Empty")

            # Laplacian-based sharpening
            edges = cv2.Laplacian(imageInput, -1)
            imageOutput = imageInput - gainFactorHighFrequencyComponent * edges
            return 0, imageOutput
        except Exception as e:
            MRCV.writeLog(f"sharpeningImage01:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1, MRCV.getErrorImage("sharpeningImage01:: Unhandled Exception")

    @staticmethod
    def sharpeningImage02(imageInput: np.ndarray, filterSize: tuple, sigmaFilter: float,
                          gainFactorHighFrequencyComponent: float) -> tuple[int, np.ndarray]:
        try:
            if imageInput.size == 0:
                return 1, MRCV.getErrorImage("sharpeningImage02:: Image is Empty")

            # Gaussian blur-based sharpening
            low = cv2.GaussianBlur(imageInput, filterSize, sigmaFilter, sigmaFilter)
            imageOutput = imageInput + gainFactorHighFrequencyComponent * (imageInput - low)
            return 0, imageOutput
        except Exception as e:
            MRCV.writeLog(f"sharpeningImage02:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1, MRCV.getErrorImage("sharpeningImage02:: Unhandled Exception")

    @staticmethod
    def readCameraParametersFromFile(pathToFileCameraParameters: str, cameraParameters: CameraParameters) -> int:
        try:
            if not os.path.exists(pathToFileCameraParameters):
                MRCV.writeLog(f"readCameraParametersFromFile:: File not found: {pathToFileCameraParameters}",
                              LOGTYPE.ERROR)
                return 2  # File not found

            fs = cv2.FileStorage(pathToFileCameraParameters, cv2.FileStorage_READ)
            if not fs.isOpened():
                MRCV.writeLog(f"readCameraParametersFromFile:: Failed to open file: {pathToFileCameraParameters}",
                              LOGTYPE.ERROR)
                fs.release()
                return 3  # Failed to open file

            # Check if required nodes exist
            required_nodes = ["M1", "D1", "R", "T", "R1", "P1", "imageSize", "rms", "avgErr"]
            for node in required_nodes:
                if fs.getNode(node).empty():
                    MRCV.writeLog(f"readCameraParametersFromFile:: Missing or invalid node: {node}", LOGTYPE.ERROR)
                    fs.release()
                    return 4  # Missing node

            cameraParameters.M1 = fs.getNode("M1").mat()
            cameraParameters.D1 = fs.getNode("D1").mat()
            cameraParameters.R = fs.getNode("R").mat()
            cameraParameters.T = fs.getNode("T").mat()
            cameraParameters.R1 = fs.getNode("R1").mat()
            cameraParameters.P1 = fs.getNode("P1").mat()
            cameraParameters.imageSize = fs.getNode("imageSize").mat()
            cameraParameters.rms = fs.getNode("rms").real()
            cameraParameters.avgErr = fs.getNode("avgErr").real()
            fs.release()

            # Calculate point maps
            M1n = np.zeros((3, 3), dtype=np.float32)
            # Ensure imageSize is a tuple of integers
            if cameraParameters.imageSize is not None:
                imageSize = tuple(int(x) for x in cameraParameters.imageSize.ravel())
                cameraParameters.map11, cameraParameters.map12 = cv2.initUndistortRectifyMap(
                    cameraParameters.M1, cameraParameters.D1, cameraParameters.R1, M1n,
                    imageSize, cv2.CV_16SC2
                )
            else:
                MRCV.writeLog("readCameraParametersFromFile:: Invalid imageSize", LOGTYPE.ERROR)
                return 5  # Invalid imageSize

            return 0
        except Exception as e:
            MRCV.writeLog(f"readCameraParametersFromFile:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1

    @staticmethod
    def preprocessingImage(image: np.ndarray, methodImagePreProcessing: list[METOD_IMAGE_PERPROCESSIN],
                           pathToFileCameraParameters: str) -> tuple[int, np.ndarray]:
        try:
            if image.size == 0:
                return 1, MRCV.getErrorImage("preprocessingImage:: Image is Empty")

            for method in methodImagePreProcessing:
                if method == METOD_IMAGE_PERPROCESSIN.NONE:
                    continue
                elif method == METOD_IMAGE_PERPROCESSIN.CONVERTING_BGR_TO_GRAY:
                    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # Grayscale image
                        pass
                    else:
                        return 2, MRCV.getErrorImage("preprocessingImage:: Unknown Image Format")
                elif method == METOD_IMAGE_PERPROCESSIN.BRIGHTNESS_LEVEL_UP:
                    status, image = MRCV.changeImageBrightness(image, 0.8)
                    MRCV.writeLog(f"\t BRIGHTNESS_LEVEL_UP, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BRIGHTNESS_LEVEL_DOWN:
                    status, image = MRCV.changeImageBrightness(image, 1.25)
                    MRCV.writeLog(f"\t BRIGHTNESS_LEVEL_DOWN, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST,
                                                               COLOR_MODEL.CM_YCBCR)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_02_YCBCR_CLAHE:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CLAHE,
                                                               COLOR_MODEL.CM_YCBCR, clipLimitCLAHE=2,
                                                               gridSizeCLAHE=(8, 8))
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_02_YCBCR_CLAHE, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING,
                                                               COLOR_MODEL.CM_YCBCR, percentContrastBalance=3)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION,
                                                               COLOR_MODEL.CM_YCBCR, mContrastExtantion=-1,
                                                               eContrastExtantion=2)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_05_HSV_EQUALIZEHIST:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST,
                                                               COLOR_MODEL.CM_HSV)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_05_HSV_EQUALIZEHIST, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_06_HSV_CLAHE:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CLAHE,
                                                               COLOR_MODEL.CM_HSV, clipLimitCLAHE=2,
                                                               gridSizeCLAHE=(8, 8))
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_06_HSV_CLAHE, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING,
                                                               COLOR_MODEL.CM_HSV, percentContrastBalance=3)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION,
                                                               COLOR_MODEL.CM_HSV, mContrastExtantion=-1,
                                                               eContrastExtantion=2)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_09_LAB_EQUALIZEHIST:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST,
                                                               COLOR_MODEL.CM_LAB)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_09_LAB_EQUALIZEHIST, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_10_LAB_CLAHE:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CLAHE,
                                                               COLOR_MODEL.CM_LAB, clipLimitCLAHE=2,
                                                               gridSizeCLAHE=(8, 8))
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_10_LAB_CLAHE, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING,
                                                               COLOR_MODEL.CM_LAB, percentContrastBalance=3)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION,
                                                               COLOR_MODEL.CM_LAB, mContrastExtantion=-1,
                                                               eContrastExtantion=2)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_13_RGB_EQUALIZEHIST:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST,
                                                               COLOR_MODEL.CM_RGB)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_13_RGB_EQUALIZEHIST, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_14_RGB_CLAHE:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CLAHE,
                                                               COLOR_MODEL.CM_RGB, clipLimitCLAHE=2,
                                                               gridSizeCLAHE=(8, 8))
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_14_RGB_CLAHE, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING,
                                                               COLOR_MODEL.CM_RGB, percentContrastBalance=3)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION:
                    status, image = MRCV.increaseImageContrast(image, METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION,
                                                               COLOR_MODEL.CM_RGB, mContrastExtantion=-1,
                                                               eContrastExtantion=2)
                    MRCV.writeLog(f"\t BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.SHARPENING_01:
                    status, image = MRCV.sharpeningImage01(image, 2)
                    MRCV.writeLog(f"\t SHARPENING_01, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.SHARPENING_02:
                    status, image = MRCV.sharpeningImage02(image, (9, 9), 0, 4)
                    MRCV.writeLog(f"\t SHARPENING_02, state = {status}")
                elif method == METOD_IMAGE_PERPROCESSIN.NOISE_FILTERING_01_MEDIAN_FILTER:
                    image = cv2.medianBlur(image, 3)
                    MRCV.writeLog("\t NOISE_FILTERING_01_MEDIAN_FILTER, state = 0")
                elif method == METOD_IMAGE_PERPROCESSIN.NOISE_FILTERING_02_AVARAGE_FILTER:
                    image = cv2.GaussianBlur(image, (3, 3), 0, 0)
                    MRCV.writeLog("\t NOISE_FILTERING_02_AVARAGE_FILTER, state = 0")
                elif method == METOD_IMAGE_PERPROCESSIN.CORRECTION_GEOMETRIC_DEFORMATION:
                    cameraParameters = CameraParameters()
                    status = MRCV.readCameraParametersFromFile(pathToFileCameraParameters, cameraParameters)
                    if status != 0:
                        MRCV.writeLog(f"\t CORRECTION_GEOMETRIC_DEFORMATION, state = {status}")
                        return status, image
                    if cameraParameters.map11 is None or cameraParameters.map12 is None:
                        MRCV.writeLog("\t CORRECTION_GEOMETRIC_DEFORMATION, map11 or map12 is None", LOGTYPE.ERROR)
                        return 6, MRCV.getErrorImage("preprocessingImage:: Invalid mapping arrays")
                    image = cv2.remap(image, cameraParameters.map11, cameraParameters.map12, cv2.INTER_LINEAR)
                    MRCV.writeLog(f"\t CORRECTION_GEOMETRIC_DEFORMATION, state = {status}")

            return 0, image
        except Exception as e:
            MRCV.writeLog(f"preprocessingImage:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1, MRCV.getErrorImage("preprocessingImage:: Unhandled Exception")

    @staticmethod
    def increaseImageContrast(imageInput: np.ndarray, methodIncreaseContrast: METOD_INCREASE_IMAGE_CONTRAST,
                              colorSpace: COLOR_MODEL, clipLimitCLAHE: float = 0, gridSizeCLAHE: tuple = (0, 0),
                              percentContrastBalance: float = 0, mContrastExtantion: float = 0,
                              eContrastExtantion: float = 0) -> tuple[int, np.ndarray]:
        try:
            if imageInput.size == 0:
                return 1, MRCV.getErrorImage("increaseImageContrast:: Image is Empty")

            if len(imageInput.shape) == 3 and imageInput.shape[2] == 3:  # Color image
                if colorSpace == COLOR_MODEL.CM_HSV:
                    qc = 2  # Value channel
                    imageOtherModel = cv2.cvtColor(imageInput, cv2.COLOR_BGR2HSV)
                    planes = list(cv2.split(imageOtherModel))  # Convert tuple to list

                    if methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST:
                        planes[qc] = cv2.equalizeHist(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CLAHE:
                        clahe = cv2.createCLAHE(clipLimit=clipLimitCLAHE, tileGridSize=gridSizeCLAHE)
                        planes[qc] = clahe.apply(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING:
                        state = MRCV.contrastBalancing(planes[qc], percentContrastBalance)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING, state = {state}", LOGTYPE.ERROR)
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION:
                        state = MRCV.contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION, state = {state}", LOGTYPE.ERROR)

                    imageOtherModel = cv2.merge(planes)
                    imageOutput = cv2.cvtColor(imageOtherModel, cv2.COLOR_HSV2BGR)

                elif colorSpace == COLOR_MODEL.CM_LAB:
                    qc = 0  # L channel
                    imageOtherModel = cv2.cvtColor(imageInput, cv2.COLOR_BGR2Lab)
                    planes = list(cv2.split(imageOtherModel))  # Convert tuple to list

                    if methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST:
                        planes[qc] = cv2.equalizeHist(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CLAHE:
                        clahe = cv2.createCLAHE(clipLimit=clipLimitCLAHE, tileGridSize=gridSizeCLAHE)
                        planes[qc] = clahe.apply(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING:
                        state = MRCV.contrastBalancing(planes[qc], percentContrastBalance)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING, state = {state}", LOGTYPE.ERROR)
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION:
                        state = MRCV.contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION, state = {state}", LOGTYPE.ERROR)

                    imageOtherModel = cv2.merge(planes)
                    imageOutput = cv2.cvtColor(imageOtherModel, cv2.COLOR_Lab2BGR)

                elif colorSpace == COLOR_MODEL.CM_YCBCR:
                    qc = 0  # Y channel
                    imageOtherModel = cv2.cvtColor(imageInput, cv2.COLOR_BGR2YCrCb)
                    planes = list(cv2.split(imageOtherModel))  # Convert tuple to list

                    if methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST:
                        planes[qc] = cv2.equalizeHist(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CLAHE:
                        clahe = cv2.createCLAHE(clipLimit=clipLimitCLAHE, tileGridSize=gridSizeCLAHE)
                        planes[qc] = clahe.apply(planes[qc])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING:
                        state = MRCV.contrastBalancing(planes[qc], percentContrastBalance)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING, state = {state}",
                                          LOGTYPE.ERROR)
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION:
                        state = MRCV.contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion)
                        if state != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION, state = {state}",
                                          LOGTYPE.ERROR)

                    imageOtherModel = cv2.merge(planes)
                    imageOutput = cv2.cvtColor(imageOtherModel, cv2.COLOR_YCrCb2BGR)

                elif colorSpace == COLOR_MODEL.CM_RGB:
                    imageOtherModel = cv2.cvtColor(imageInput, cv2.COLOR_BGR2YCrCb)  # Using YCrCb for consistency
                    planes = list(cv2.split(imageOtherModel))  # Convert tuple to list

                    if methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST:
                        planes[0] = cv2.equalizeHist(planes[0])
                        planes[1] = cv2.equalizeHist(planes[1])
                        planes[2] = cv2.equalizeHist(planes[2])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CLAHE:
                        clahe = cv2.createCLAHE(clipLimit=clipLimitCLAHE, tileGridSize=gridSizeCLAHE)
                        planes[0] = clahe.apply(planes[0])
                        planes[1] = clahe.apply(planes[1])
                        planes[2] = clahe.apply(planes[2])
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING:
                        state0 = MRCV.contrastBalancing(planes[0], percentContrastBalance)
                        state1 = MRCV.contrastBalancing(planes[1], percentContrastBalance)
                        state2 = MRCV.contrastBalancing(planes[2], percentContrastBalance)
                        if state0 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[0] = {state0}",
                                          LOGTYPE.ERROR)
                        if state1 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[1] = {state1}",
                                          LOGTYPE.ERROR)
                        if state2 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[2] = {state2}",
                                          LOGTYPE.ERROR)
                    elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION:
                        state0 = MRCV.contrastExtantion(planes[0], mContrastExtantion, eContrastExtantion)
                        state1 = MRCV.contrastExtantion(planes[1], mContrastExtantion, eContrastExtantion)
                        state2 = MRCV.contrastExtantion(planes[2], mContrastExtantion, eContrastExtantion)
                        if state0 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[0] = {state0}",
                                          LOGTYPE.ERROR)
                        if state1 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[1] = {state1}",
                                          LOGTYPE.ERROR)
                        if state2 != 0:
                            MRCV.writeLog(f"BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[2] = {state2}",
                                          LOGTYPE.ERROR)

                    imageOtherModel = cv2.merge(planes)
                    imageOutput = cv2.cvtColor(imageOtherModel, cv2.COLOR_YCrCb2BGR)

            elif len(imageInput.shape) == 2 or (
                    len(imageInput.shape) == 3 and imageInput.shape[2] == 1):  # Grayscale image
                imageOutput = imageInput.copy()
                if methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.EQUALIZE_HIST:
                    imageOutput = cv2.equalizeHist(imageOutput)
                elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CLAHE:
                    clahe = cv2.createCLAHE(clipLimit=clipLimitCLAHE, tileGridSize=gridSizeCLAHE)
                    imageOutput = clahe.apply(imageOutput)
                elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_BALANCING:
                    state = MRCV.contrastBalancing(imageOutput, percentContrastBalance)
                    if state != 0:
                        MRCV.writeLog(f"CONTRAST_BALANCING, state = {state}", LOGTYPE.ERROR)
                elif methodIncreaseContrast == METOD_INCREASE_IMAGE_CONTRAST.CONTRAST_EXTENSION:
                    state = MRCV.contrastExtantion(imageOutput, mContrastExtantion, eContrastExtantion)
                    if state != 0:
                        MRCV.writeLog(f"CONTRAST_EXTENSION, state = {state}", LOGTYPE.ERROR)
            else:
                return 2, MRCV.getErrorImage("increaseImageContrast:: Unknown Image Format")

            return 0, imageOutput
        except Exception as e:
            MRCV.writeLog(f"increaseImageContrast:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1, MRCV.getErrorImage("increaseImageContrast:: Unhandled Exception")

    @staticmethod
    def contrastBalancing(planeArray: np.ndarray, percent: float) -> int:
        try:
            if planeArray.size == 0:
                return 1  # Empty array

            MRCV.writeLog(f"percent = {percent}", LOGTYPE.DEBUG)
            if 0 < percent < 100:
                ratio = percent / 200.0
                flat = planeArray.ravel()
                flat = np.sort(flat)
                lowValue = flat[int(np.floor(len(flat) * ratio))]
                highValue = flat[int(np.ceil(len(flat) * (1.0 - ratio)))]

                planeArray[planeArray < lowValue] = lowValue
                planeArray[planeArray > highValue] = highValue
                planeArray[:] = cv2.normalize(planeArray, None, 0, 255, cv2.NORM_MINMAX)
                return 0
            else:
                return 3  # Percent out of range
        except Exception as e:
            MRCV.writeLog(f"contrastBalancing:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1

    @staticmethod
    def contrastExtantion(planeArray: np.ndarray, m: float, e: float) -> int:
        try:
            if planeArray.size == 0:
                return 1  # Empty array

            planeDouble = planeArray.astype(np.float32) / 255.0
            if m < 0:
                m = np.mean(planeDouble)
            planeDouble = m / (planeDouble + 1e-16)
            planeDouble = np.power(planeDouble, e)
            planeDouble = 1 / (1 + planeDouble)
            planeArray[:] = (planeDouble * 255).astype(np.uint8)
            return 0
        except Exception as e:
            MRCV.writeLog(f"contrastExtantion:: Unhandled Exception: {str(e)}", LOGTYPE.ERROR)
            return -1