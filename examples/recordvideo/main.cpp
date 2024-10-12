#include <mrcv/mrcv.h>

#include <iostream>
#include <thread>

bool flag = false;

void consoleCounter()
{
    int counter = 0;
    while (true)
    {
        // Очистка экрана
#ifdef _WIN32 
        int res = std::system("cls");
#else
        int res = std::system("clear");
#endif
        // Вывод диагностической информации
        std::cout << "The video record started. Please wait ... " << std::to_string(counter++) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (flag)
        {
            std::cout << "Done...";
            return;
        }
    }
}

int main(int, char* [])
{
    // Поток записи видео
    std::thread videoThread(mrcv::recordVideo, 0, 7, "sarganCV", mrcv::CODEC::XVID);
    // Поток вывода в консоль
    std::thread counterThread(consoleCounter);

    videoThread.join();
    flag = true;
    counterThread.join();

    std::cin.get();
}