import numpy as np
import os
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DenseStereo:
    class Vuxyzrgb:
        def __init__(self):
            self.xyz = []  # Список для хранения координат точек

    def __init__(self):
        self.vuxyzrgb = self.Vuxyzrgb()
        self.IDX = []  # Вектор индексов кластеров

    def load_data_from_file(self, filename: str) -> None:
        """
        Загружает данные из указанного файла.

        Args:
            filename: Имя файла, из которого будут загружены данные.
        """
        if not os.path.exists(filename):
            logger.error(f"Файл не найден: {filename}")
            return

        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Очищаем предыдущие данные
                self.vuxyzrgb.xyz.clear()
                line_count = 0
                for line in file:
                    line_count += 1
                    data = line.strip().split()
                    if len(data) < 8:
                        logger.warning(f"Некорректная строка {line_count} в файле {filename}: {line.strip()}")
                        continue
                    # Считываем координаты (x, y, z) из колонок 2, 3, 4
                    try:
                        point = [float(data[2]), float(data[3]), float(data[4])]
                        self.vuxyzrgb.xyz.append(point)
                    except ValueError as e:
                        logger.warning(f"Ошибка преобразования в строке {line_count}: {line.strip()}. Ошибка: {e}")
                logger.info(f"Прочитано {line_count} строк, загружено {len(self.vuxyzrgb.xyz)} точек из файла {filename}")
                if len(self.vuxyzrgb.xyz) != line_count:
                    logger.warning(f"Не все строки загружены: прочитано {line_count}, загружено {len(self.vuxyzrgb.xyz)}")
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {filename}: {e}")

    @staticmethod
    def compute_distances(args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Вычисляет расстояния между точками в 3D-пространстве.

        Args:
            args: Кортеж с параметрами (start, end, x3D, y3D, z3D).

        Returns:
            Кортеж с матрицей расстояний и суммой расстояний.
        """
        start, end, x3D, y3D, z3D = args
        n = len(x3D)
        S = np.zeros((end - start, n))
        sumS = 0.0
        for i in range(start, end):
            S_i = np.sqrt((x3D[i] - x3D) ** 2 + (y3D[i] - y3D) ** 2 + (z3D[i] - z3D) ** 2)
            sumS += S_i.sum()
            S[i - start] = S_i
        logger.debug(f"Вычислены расстояния для блока {start}:{end}")
        return S, sumS

    @staticmethod
    def normalize_distances(args: Tuple[int, int, np.ndarray, float]) -> np.ndarray:
        """
        Нормализует расстояния.

        Args:
            args: Кортеж с параметрами (start, end, S, S2).

        Returns:
            Нормализованная матрица расстояний.
        """
        start, end, S, S2 = args
        S3 = S[start:end] / S2
        logger.debug(f"Нормализованы расстояния для блока {start}:{end}")
        return S3

    @staticmethod
    def compute_proximity(args: Tuple[int, int, np.ndarray, float]) -> np.ndarray:
        """
        Вычисляет близость точек.

        Args:
            args: Кортеж с параметрами (start, end, S3, coef).

        Returns:
            Матрица близости (булева).
        """
        start, end, S3, coef = args
        L = S3[start:end] < coef
        logger.debug(f"Вычислена близость для блока {start}:{end}")
        return L

    def make_clustering(self) -> None:
        """
        Выполняет кластеризацию загруженных данных.
        """
        if not self.vuxyzrgb.xyz:
            logger.warning("Нет данных для кластеризации. Убедитесь, что данные загружены через load_data_from_file.")
            return

        try:
            # Инициализация
            coef = 0.1
            xyz = np.array(self.vuxyzrgb.xyz)
            x3D, y3D, z3D = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            Len3D = len(xyz)
            logger.info(f"Начало кластеризации. Количество точек: {Len3D}")

            # Создание матрицы расстояний
            num_processes = cpu_count()
            # Динамическое разбиение на блоки
            block_size = (Len3D + num_processes - 1) // num_processes  # Округление вверх
            tasks = [(i * block_size, min((i + 1) * block_size, Len3D), x3D, y3D, z3D)
                     for i in range((Len3D + block_size - 1) // block_size)]

            with Pool(num_processes) as pool:
                results = pool.map(self.compute_distances, tasks)

            S = np.vstack([result[0] for result in results])
            sumS = sum(result[1] for result in results)
            S2 = sumS / (Len3D * Len3D)
            logger.info(f"Матрица расстояний создана, размер: {S.shape}")
            if S.shape != (Len3D, Len3D):
                logger.error(f"Некорректный размер матрицы S: ожидалось ({Len3D}, {Len3D}), получено {S.shape}")
                raise ValueError("Ошибка в размере матрицы S")

            # Нормализация расстояний
            tasks = [(i * block_size, min((i + 1) * block_size, Len3D), S, S2)
                     for i in range((Len3D + block_size - 1) // block_size)]
            with Pool(num_processes) as pool:
                S3 = np.vstack(pool.map(self.normalize_distances, tasks))
            logger.info(f"Расстояния нормализованы, размер S3: {S3.shape}")
            if S3.shape != (Len3D, Len3D):
                logger.error(f"Некорректный размер матрицы S3: ожидалось ({Len3D}, {Len3D}), получено {S3.shape}")
                raise ValueError("Ошибка в размере матрицы S3")

            # Вычисление матрицы близости
            tasks = [(i * block_size, min((i + 1) * block_size, Len3D), S3, coef)
                     for i in range((Len3D + block_size - 1) // block_size)]
            with Pool(num_processes) as pool:
                L = np.vstack(pool.map(self.compute_proximity, tasks))
            logger.info(f"Матрица близости вычислена, размер L: {L.shape}")
            if L.shape != (Len3D, Len3D):
                logger.error(f"Некорректный размер матрицы L: ожидалось ({Len3D}, {Len3D}), получено {L.shape}")
                raise ValueError("Ошибка в размере матрицы L")

            # Кластеризация
            self.IDX = np.full(Len3D, -1, dtype=int)
            clusters_data = []
            num_cluster = 0
            IDs_check = [0]
            IDs_clusters = [i for i in range(Len3D) if L[0, i]]
            IDs_lost = [i for i in range(Len3D) if not L[0, i]]

            while IDs_lost:
                ID_current = IDs_lost[0]
                IDs_clusters = [i for i in range(Len3D) if L[ID_current, i]]

                VectDiff = [i for i in IDs_clusters if i not in IDs_check]
                while VectDiff:
                    ID_current = VectDiff[0]
                    new_points = [i for i in range(Len3D) if L[ID_current, i] and i not in IDs_clusters]
                    IDs_clusters.extend(new_points)
                    if ID_current not in IDs_check:
                        IDs_check.append(ID_current)
                    VectDiff = [i for i in IDs_clusters if i not in IDs_check]

                IDs_lost = [i for i in IDs_lost if i not in IDs_clusters]
                xyz2 = xyz[IDs_clusters].tolist()
                for i in IDs_clusters:
                    self.IDX[i] = num_cluster
                num_cluster += 1
                clusters_data.append(xyz2)

            # Вывод количества кластеров
            logger.info(f"Количество кластеров: {num_cluster}")

            # Запись кластеров и точек в файл
            os.makedirs('files', exist_ok=True)
            with open('files/clusters_data.txt', 'w', encoding='utf-8') as f:
                f.write("Координаты точки, номер точки, номер кластера\n")
                for i in range(Len3D):
                    f.write(f"{x3D[i]}, {y3D[i]}, {z3D[i]}, {i}, {self.IDX[i]}\n")
            logger.info(f"Результаты кластеризации сохранены в files/clusters_data.txt")

        except Exception as e:
            logger.error(f"Ошибка в процессе кластеризации: {e}")
            raise

    def get_num_clusters(self) -> int:
        """
        Возвращает количество кластеров.

        Returns:
            Количество кластеров.
        """
        num_clusters = max(self.IDX) + 1 if self.IDX.size > 0 and max(self.IDX) >= 0 else 0
        return num_clusters

    def print_clusters(self) -> None:
        """
        Печатает информацию о кластерах.
        """
        num_points = len(self.IDX)
        num_clusters = self.get_num_clusters()
        logger.info(f"Кластеризация выполнена успешно. Количество точек: {num_points}, "
                    f"количество кластеров: {num_clusters}")