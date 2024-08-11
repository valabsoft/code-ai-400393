#pragma once
#ifndef MRCV_CLUSTERING_H
#define MRCV_CLUSTERING_H

#include <vector>
#include <mutex>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <thread>
#include <algorithm>
#include <random>

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkNamedColors.h>
#include <vtkProperty.h>

namespace mrcv {

    /**
     * @brief Структура для работы с плотным стерео и кластеризацией.
     *
     * Структура `DenseStereo` предоставляет функционал для загрузки данных,
     * выполнения кластеризации, вывода и визуализации кластеров.
     */
    struct DenseStereo {

        /**
         * @brief Выполняет кластеризацию загруженных данных.
         *
         * Функция для выполнения кластеризации данных, хранящихся
         * в `vuxyzrgb`. Результаты кластеризации сохраняются в `IDX`.
         */
        void Clustering();

        /**
         * @brief Загружает данные из файла.
         *
         * Функция считывает данные из указанного файла и сохраняет их
         * во внутренней структуре `vuxyzrgb`.
         *
         * @param filename Имя файла, из которого будут загружены данные.
         */
        void loadDataFromFile(const std::string& filename);

        /**
         * @brief Печатает информацию о кластерах.
         *
         * Функция выводит на экран информацию о кластерах,
         * сформированных в результате выполнения кластеризации.
         */
        void printClusters();

        /**
         * @brief Визуализирует кластеры в 3D.
         *
         * Функция отображает результаты кластеризации в 3D пространстве,
         * используя данные из `vuxyzrgb`.
         */
        void visualizeClusters3D();

    private:
        /**
         * @brief Структура для хранения координат точек.
         *
         * В этой структуре сохраняются трехмерные координаты точек,
         * используемых в процессе кластеризации.
         */
        struct {
            std::vector<std::vector<double>> xyz; ///< Трехмерные координаты точек.
        } vuxyzrgb;

        std::mutex vuxyzrgb_mutex; ///< Мьютекс для защиты данных `vuxyzrgb`.

        std::vector<int> IDX; ///< Вектор индексов кластеров для каждой точки.
    };

} // namespace mrcv

#endif // MRCV_CLUSTERING_H
