#include <mrcv/mrcv.h>

namespace mrcv
{
	// функция чтения файла и сохранения в строку    
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

	// вспомогательная функция поиска подстроки
	// VA: Переписать заголовок в виде
	// int findSubstring(std::string text, std::string word, std::vector<int>& indexes)
	std::vector<int> findSubstring(std::string text, std::string word)
	{
		std::vector<int> substringAllPositions;
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

		return substringAllPositions;
	}

	// поиск url картинок в тексте файла
	std::vector<std::string> findUrl(std::string text)
	{
		int ind = 0;
		std::vector<int> positionAllTegHttps;   // список всех позиций слова https
		std::string wordBegin{ "https" };        // слово для поиска
		std::vector<int> positionAllTegJpg;     // список всех позиций слова jpg
		std::string wordEnd{ ".jpg" };           // слово для поиска
		std::vector<int> positionTegHttps;      // список синхронизированных позиций слова https
		std::vector<int> positionTegJpg;        // список синхронизированных позиций слова jpg
		std::vector<std::string> arrUrl;        // список найденных url

		positionAllTegHttps = findSubstring(text, wordBegin); // поиск позиций всех слов https в тексте
		positionAllTegJpg = findSubstring(text, wordEnd);     // поиск позиций всех слов .jpg в тексте

		// В тексте не нашлось ниодной ссылки        
		if (positionAllTegHttps.size() == 0)
		{
			arrUrl.push_back("0");
			return arrUrl;
		}
		if (positionAllTegJpg.size() == 0)
		{
			arrUrl.push_back("0");
			return arrUrl;
		}
		// цикл синхронизации тегов https и .jpg, отбрасываем все номера позиций https которые не заканчиваюится на  .jpg
		for (unsigned long i = 0; i < positionAllTegHttps.size() - 1; i++)
		{
			if (positionAllTegJpg[ind] > positionAllTegHttps[i + 1])
			{
			}
			else
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
		return arrUrl;
	}

	// качаем фото из списка, возвращаем последний индекс файла
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
			std::string str1 = "wget   --tries=2  --connect-timeout=5 ";
			// тихая консоль, добавлен параметр -q который ничего не выводит в консоль
			// std::string str1 = "wget  -r  --tries=2  --connect-timeout=5  -q "; // VA - Не используется?
			std::string str2 = " --user-agent=\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36\" ";
			std::string concat_str = str1 + str2 + " -O " + path + fileName + namestep + ".jpg ";
			std::string cmd = concat_str + arrUrl[i];
			// вызов исполняемой функции
			system(cmd.c_str());
		}
		return index; // возвращаем последний индекс
	}

	void deleleSmall(std::string pathToFile, int rows, int cols)
	{
		std::string path = pathToFile;
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
					std::cout << e.what();
				}
			}
		}
	}
	// копируем файлы в папки train и test
	void copyFile(std::string filePath, std::string target, int percent)
	{
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
					std::cout << e.what();
				}
			}
		}
	}

	// функция скачаивания файла Яндекса
	void getFileFromYandex(std::string zapros, bool money, int pageInt, std::string key = "", std::string secretKey = "")
	{
		zapros += "\" ";
		std::string page = std::to_string(pageInt);
		std::string strzapros;
		std::string str1 = "curl -X GET -o result.xml ";
		std::string str1free = "curl -X GET -o result.xml  -A \"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36\" \"https://yandex.ru/images/search?from=tabbar&text=";
		std::string str2 = " \"https://yandex.ru/images-xml?folderid=";
		std::string str3 = "&apikey=";
		std::string str4 = "&p=";
		std::string str5 = "&text=";

		if (money)
			strzapros = str1 + str2 + key + str3 + secretKey + str4 + page + str5 + zapros;
		else
			strzapros = str1free + zapros;

		std::string cmd = strzapros;
		std::cout << cmd << '\n';
		system(cmd.c_str());
	}

	// основная функция
	int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey)
	{
		if (trainsetPercentage < 5 || trainsetPercentage > 90)
			return 1501;
		
		if ((money) && ((key == "") || (secretKey == "")))
			return 1502;
		
		if (minWidth < 100 || minWidth > 1900)
			return 1503;
		
		if (minHeight < 100 || minHeight > 1900)
			return 1504;
		
		if (nameTemplate == "")
			return 1505;
		
		if (countFoto <= 5 || countFoto >= 500)
			return 1506;
		
		if (!(std::filesystem::exists(outputFolder)))
			return 1507;

		int page = -1;
		unsigned int countFotoindDir = 0; // количество изображений в папке, изначально 0
		int index = 0; // техническая переменная для защиты от бесконечного цикла
		while (countFotoindDir < countFoto) // если количество фото в папке меньше чем надо скачать цикл продолжается
		{
			page = page + 1; // страница поиска, изначально 0
			if ((money == false) && (page == 1)) // условие выхода из цикла если используется бесплатный режим, у бесплатного режима страница поиска только 0
				break;
			getFileFromYandex(query, money, page, key, secretKey); // качаем страницу Яндекса в файл "result.xml"


			if (!std::filesystem::exists("result.xml"))
				return 1508;

			if (std::filesystem::file_size("result.xml") < 100)
				return 1509;

			std::string text = readFile("result.xml"); // читаем  файл
			std::vector<std::string> arrUrl;
			arrUrl = findUrl(text); //поиск в скаченном файле url

			if (arrUrl[0] == "0")
				return 1510;

			countFotoindDir = downloadFoto(arrUrl, outputFolder, nameTemplate, countFotoindDir, countFoto); // качаем все фото из списка, возвращаем последний индекс скачанного файла
			if (countFotoindDir == 0)
				return 1511;

			index = index + 1;
			if (index == 20) // защита от бесконечного цикла, если условия цикла while никогда не станет false, сработает это условие через 20 итераций
				return 1512;
		}

		deleleSmall(outputFolder, minWidth, minHeight); // удаляем мелкие фото (ширина, высота) и фото с нулевым размером
		if (separateDataset)
			copyFile(outputFolder, outputFolder, trainsetPercentage); // раскладываем файлы по папкам в процентном отношении 70% 

		return 1500;
	}
}