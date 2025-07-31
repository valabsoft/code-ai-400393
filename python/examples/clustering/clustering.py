from mrcv import DenseStereo
import logging

data_path = "../../../examples/clustering/files/claster.dat"

def main():
    """
    Основная функция для выполнения кластеризации.
    """
    dense_stereo = DenseStereo()  # Замените на актуальную инициализацию
    dense_stereo.load_data_from_file(data_path)
    dense_stereo.make_clustering()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Для поддержки замороженных исполняемых файлов
    main()