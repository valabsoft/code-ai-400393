#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <list>
#include <string>
#include <sstream>
#include <vector>

#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
    /**
    * @brief Вычисление расстояния между двумя 3D точками
    * @param p1 - первая точка
    * @param p2 - вторая точка
    * @return distance - расстояние между точками
    */
    double geometryGetDistance(Point3D p1, Point3D p2);

    /**
    * @brief Расчет количества точек в окресности заданной габаритной точки
    * @param std::vector<Cloud3DItem> cloud3D - облако 3D-точек объекта
    * @param std::vector<double> X - координаты X облака точек объекта
    * @param std::vector<double> Y - координаты Y облака точек объекта
    * @param std::vector<double> Z - координаты Z облака точек объекта
    * @param Point3D MN - заданная габаритной точка
    * @param Point3D M0 - центр масс объекта
    * @return size_t - количество точек в окресности
    */
    size_t geometryGetNumberOfNearestPoints(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, Point3D MN, Point3D M0);
    
    double geometryLineSegDistance(std::vector<double> vertexP, std::vector<double> vertexP0, std::vector<double> vertexP1);
    std::vector<double> geomentryVectorSubtraction(std::vector<double> A, std::vector<double> B);
    double geometryVectorNorm(std::vector<double> A);
    std::vector<double> geometryVectroCross(std::vector<double> A, std::vector<double> B);
}
