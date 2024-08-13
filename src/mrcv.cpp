#include <mrcv/mrcv.h>

namespace mrcv
{
    /**
     * @brief Функция сложения двух целых чисел.
     * @param a - Первое слагаемое.
     * @param b - Второе слагаемое.
     * @return - Резальтат вычсиления выражения a + b
     */
    int add(int a, int b)
    {
        return a + b;
    }

    MRCVPoint::MRCVPoint()
    {
        _X = 0;
        _Y = 0;
    }
    void MRCVPoint::setX(int X)
    {
        _X = X;
    }
    void MRCVPoint::setY(int Y)
    {
        _Y = Y;
    }
    std::string MRCVPoint::gerCoordinates()
    {
        return "(" + std::to_string(_X) + ";" + std::to_string(_Y) + ")";
    }
}
