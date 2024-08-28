#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    auto sum = mrcv::add(3, 2);
    
    ///////////////////////////////////////////////////////////////////////////
    // Вывод в консоль вида std::cout << ...
    // нужно или убрать, или заменить на вывод в диагностический лог mrcv::writeLog(...)
    ///////////////////////////////////////////////////////////////////////////

    // Старый формат
    std::cout << "3 + 2 = " << sum << std::endl;
    
    // Новый формат
    mrcv::writeLog("3 + 2 = " + std::to_string(sum), mrcv::LOGTYPE::INFO);    
    
    return 0;
}