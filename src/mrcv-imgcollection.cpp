#include <mrcv/mrcv.h>

namespace mrcv
{
  /**
  * @brief функция чтения файла и сохранения в строку.
  * @param fileName - Имя файла.
  * @return - содержимое файла в формате string.
  */
  std::string readFile(const std::string& fileName)
  {
    std::ifstream f(fileName);
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    std::string s(size, ' ');
    f.seekg(0);
    f.read(&s[0], size);
    return s;
  }
  
  /**
  * @brief функция поиска подстроки.
  * @param text - текст в котором будет осуществен поиск.
  * @param word - строка поиска.
  * @param substringAllPositions - вектор найденных позиций.
  * @return - результат работы функции.
  */
  int findSubstring(std::string text, std::string word, std::vector<int>& substringAllPositions)
  {
    for (size_t i{}; i <= text.length() - word.length(); )
    {
      // получаем индекс
      size_t substringPosition = text.find(word, i);
      // если не найдено ни одного вхождения с индекса i, выходим из цикла
      if (substringPosition == std::string::npos)
       break;

      substringAllPositions.push_back((int)substringPosition);
      // если же вхождение найдено, увеличиваем счетчик вхождений
      // переходим к следующему индексу после positionSubstring
      i = substringPosition + 1;
    }
    if (substringAllPositions.size() == 0)
      return 10;
    return 0;
  }
  
  /**
  * @brief функция поиска url картинок в тексте файла.
  * @param text - текст в котором будет осуществен поиск.
  * @param arrUrl - вектор адресов изображений.
  * @return - результат работы функции.
  */
  int findUrl(std::string text,std::vector<std::string>& arrUrl)
  {
    int ind = 0;
    std::vector<int> positionAllTegHttps;   // список всех позиций слова https
    std::string wordBegin{ "https" };        // слово для поиска
    std::vector<int> positionAllTegJpg;     // список всех позиций слова jpg
    std::string wordEnd{ ".jpg" };           // слово для поиска
    std::vector<int> positionTegHttps;      // список синхронизированных позиций слова https
    std::vector<int> positionTegJpg;        // список синхронизированных позиций слова jpg

    int resultFindSubstring = findSubstring(text, wordBegin,positionAllTegHttps); // поиск позиций всех слов https в тексте
    // В тексте не нашлось ниодной ссылки 
    if (resultFindSubstring != 0)
      return resultFindSubstring;
    resultFindSubstring = findSubstring(text, wordEnd,positionAllTegJpg);     // поиск позиций всех слов .jpg в тексте
      if (resultFindSubstring != 0)
        return resultFindSubstring;
       
    // цикл синхронизации тегов https и .jpg, отбрасываем все номера позиций https которые не заканчиваюится на  .jpg
    for (unsigned long i = 0; i < positionAllTegHttps.size() - 1; i++)
    {
      if (positionAllTegJpg[ind] < positionAllTegHttps[i + 1])
      {
        positionTegHttps.push_back(positionAllTegHttps[i]);
        positionTegJpg.push_back(positionAllTegJpg[ind]);
        ++ind;
      }
    }
    // цикл поиска подстроки url  в тексте и сохранения их в std::vector< std::string > arrUrl;
    for (unsigned long i = 0; i < positionTegHttps.size() - 1; i++)
    {
      if (positionTegJpg[i] - positionTegHttps[i] + 4 > 0)
      {
        int lengthUrl = positionTegJpg[i] - positionTegHttps[i] + 4; // длина слова составляющая url
        std::string strA = text.substr(positionTegHttps[i], lengthUrl); //вырезаем url из общего текста
        strA = std::regex_replace(strA, std::regex("%3A%2F%2F"), "://"); // исправляем кодировку
        strA = std::regex_replace(strA, std::regex("%2F"), "/"); // исправляем кодировку
        arrUrl.push_back(strA); // сохраняем все найденные url
      }
    }
    return 0;
  }

  /**
  * @brief функция закачки фото из списка.
  * @param arrUrl - вектор адресов изображений.
  * @param path - путь куда сохраняются изображения.
  * @param fileName - имя файла изображений.
  * @param countFotoindDir - количество уже скачанных изображений.
  * @param countFoto - количество изображений которое необходимо скачать.
  * @return - возвращает последний индекс файла.
  */
  unsigned int downloadFoto(std::vector<std::string> arrUrl, std::string path, std::string fileName, unsigned int countFotoindDir, unsigned int countFoto)
  {
    unsigned int index = 0;
    for (unsigned int i = 0; i < arrUrl.size() - 1; i++)
    {
      index = i + countFotoindDir; // цифровой индекс для добавлении к имени файла
      if (countFoto < i) // выход из цикла если скачали достаточное количество
        break;

      std::string namestep = std::to_string(index); // порядковая цифра к названию файла
      // собираем строку
      // если не надо выводить в консоль, добавить параметр -q
      std::string wgetString1 = "wget --tries=2 --connect-timeout=5 ";
      std::string wgetString2 = " --user-agent=\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36\" ";
      std::string concat_str = wgetString1 + wgetString2 + " -O " + path + fileName + namestep + ".jpg ";
      std::string cmd = concat_str + arrUrl[i];
      // вызов исполняемой функции
      system(cmd.c_str());
    }
    return index; // возвращаем последний индекс
  }

  /**
  * @brief функция удаления фото по не соответсвию размера по ширине и высоте и фото с нулевым размером.
  * @param pathToFile - путь к папке.
  * @param rows - минимальная ширина.
  * @param cols - минимальная высота.
  * @return - результат работы функции.
  */
  int deleleSmall(std::string pathToFile, int rows, int cols)
  {
    std::string path = pathToFile;
    int index = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {      
      if (entry.is_regular_file())
      {
        cv::Mat src = cv::imread(entry.path().string());
        // удаляем файлы по условию - ширина и высота фото и размер 0 байт
        if ((src.rows <= rows && src.cols <= cols) || (std::filesystem::file_size(entry.path()) == 0))
        try
        {
          remove(entry.path());
        }
        catch (std::exception& e)
        {
          index = 13;
          std::cout << e.what();
        }
      }
    }
    return index;
  }
  
  /**
  * @brief функция копирует файлы в папки train и test.
  * @param filePath - путь к папке из какой папки берем фото.
  * @param target - путь к папке в какую папку копируем.
  * @param percent - процент.
  * @return - результат работы функции.
  */
  int copyFile(std::string filePath, std::string target, int percent)
  {
    int index=0;
    std::string path = filePath; // из какой папки берем фото
    std::string train = "train";
    std::string test = "test";
    std::string pathTrain = target; // в какую папку копируем
    std::string pathTest = target; // в какую папку копируем
    pathTrain += train; // добавляем в конечную папку, папку  train
    pathTest += test; // добавляем в конечную папку, папку  test
    std::filesystem::path targetPath;
    std::string pathRez;
    
    if (!(std::filesystem::exists(pathTrain)))
      std::filesystem::create_directories(pathTrain);
    if (!(std::filesystem::exists(pathTest)))
      std::filesystem::create_directories(pathTest);

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
      if (entry.is_regular_file())
      {
        std::filesystem::path sourceFile = entry.path();

        if ((rand() % 100) < percent)
          pathRez = pathTrain;
        else
          pathRez = pathTest;

        std::filesystem::path targetParent = pathRez;
        targetPath = targetParent / sourceFile.filename();
        try
        {
          std::filesystem::copy_file(sourceFile, targetPath, std::filesystem::copy_options::overwrite_existing);
        }
        catch (std::exception& e)
        {
          index = 14;
          std::cout << e.what();
        }
      }
    }
    return index;
  }

  /**
  * @brief функция функция скачаивания файла Яндекса.
  * @param filePath - путь к папке из какой папки берем фото.
  * @param target - путь к папке в какую папку копируем.
  * @param percent - процент.
  * @return - результат работы функции.
  */
  int getFileFromYandex(std::string queryString, bool money, int pageInt, std::string key = "", std::string secretKey = "")
  {
    queryString += "\" ";
    std::string page = std::to_string(pageInt);
    std::string strqueryString;
    std::string strCurl1 = "curl -X GET -o result.xml ";
    std::string strFree = "curl -X GET -o result.xml -A \"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36\" \"https://yandex.ru/images/search?from=tabbar&text=";
    std::string strCurl2 = " \"https://yandex.ru/images-xml?folderid=";
    std::string strCurl3 = "&apikey=";
    std::string strCurl4 = "&p=";
    std::string strCurl5 = "&text=";

    if (money)
      strqueryString = strCurl1 + strCurl2 + key + strCurl3 + secretKey + strCurl4 + page + strCurl5 + queryString;
    else
      strqueryString = strFree + queryString;

    std::string cmd = strqueryString;
    std::cout << cmd << '\n';
    system(cmd.c_str());
    
    if (!std::filesystem::exists("result.xml"))
      return 8;
    return 0;
  }
  /**
    * @brief Функция скачаивания файла Яндекса.
    * @param query - Строка запроса для поиска.
    * @param minWidth - Минимальная ширина изображения.
    * @param minHeight - Минимальная высота изображения.
    * @param nameTemplate - Шаблон имени файла.
    * @param outputFolder - Папка для скачивания.
    * @param separateDataset - Флаг разбивки датасета на тренировочную и тестовую выборки.
    * @param trainsetPercentage - Процент для распределения между папками.
    * @param countFoto - Количество необходимых фото для скачивания.
    * @param money - Платный или бесплатный вариант работы.
    * @param key - Яндекс key.
    * @param secretKey - Яндекс secretKey.
    * @return - Результат работы функции.
    */
  int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey)
  {
    if (trainsetPercentage < 5 || trainsetPercentage > 90)
      return 1;
    
    if ((money) && ((key == "") || (secretKey == "")))
      return 2;
    
    if (minWidth < 100 || minWidth > 1900)
      return 3;
    
    if (minHeight < 100 || minHeight > 1900)
      return 4;
    
    if (nameTemplate == "")
      return 5;
    
    if (countFoto <= 5 || countFoto >= 500)
      return 6;
    
    if (!(std::filesystem::exists(outputFolder)))
      return 7;

    int page = -1;
    unsigned int countFotoindDir = 0; // количество изображений в папке, изначально 0
    int index = 0; // техническая переменная для защиты от бесконечного цикла
    while (countFotoindDir < countFoto) // если количество фото в папке меньше чем надо скачать цикл продолжается
    {
      page = page + 1; // страница поиска, изначально 0
      if ((money == false) && (page == 1)) // условие выхода из цикла если используется бесплатный режим, у бесплатного режима страница поиска только 0
        break;
      
      int resultFileYandex = getFileFromYandex(query, money, page, key, secretKey); // качаем страницу Яндекса в файл "result.xml"
      if (resultFileYandex != 0)
        return 8;

      if (std::filesystem::file_size("result.xml") < 100)
        return 9;

      std::string text = readFile("result.xml"); // читаем  файл
      std::vector<std::string> arrUrl;
      int kodFindUrl = findUrl(text,arrUrl); //поиск в скаченном файле url

      if (kodFindUrl != 0)
        return 10;

      countFotoindDir = downloadFoto(arrUrl, outputFolder, nameTemplate, countFotoindDir, countFoto); // качаем все фото из списка, возвращаем последний индекс скачанного файла
      if (countFotoindDir == 0)
        return 11;

      index = index + 1;
      if (index == 20) // защита от бесконечного цикла, если условия цикла while никогда не станет false, сработает это условие через 20 итераций
        return 12;
    }

    int resuiltDelete = deleleSmall(outputFolder, minWidth, minHeight); // удаляем мелкие фото (ширина, высота) и фото с нулевым размером
    
    if (separateDataset)
      int resuiltCopy = copyFile(outputFolder, outputFolder, trainsetPercentage); // раскладываем файлы по папкам в процентном отношении 70% 

    return 0;
  }
}
