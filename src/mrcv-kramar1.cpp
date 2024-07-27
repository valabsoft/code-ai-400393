#include <mrcv/mrcv.h>

namespace mrcv
{
   
	
// static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
// {
//     ((std::string*)userp)->append((char*)contents, size * nmemb);
//     return size * nmemb;
// }
// std::string  parserUrlString(const char* nameFind)
// {
//     CURL *curl;
//       CURLcode res;
//       std::string readBuffer;
//     curl = curl_easy_init();
// 
//       char* esc_text= curl_easy_escape( curl, nameFind, 0);
//       std::string url=  "https://yandex.ru/images/search?from=tabbar&text=";
//       url+= esc_text;
// 
//       if(curl) {
//         curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//         curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
//         curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
//         res = curl_easy_perform(curl);
//         curl_easy_cleanup(curl);
//       }
//       return readBuffer;
// }
		
		
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


	//вспомогательная функция поиска подстроки
	std::vector< int > findSubstring(std::string text,std::string word)
	{
        std::vector< int > positionAll;
	    for (unsigned i {}; i <= text.length() - word.length(); )
            {
                // получаем индекс
                size_t position = text.find(word, i);
                // если не найдено ни одного вхождения с индекса i, выходим из цикла
                if (position == std::string::npos) break;
                // если же вхождение найдено, увеличиваем счетчик вхождений
                positionAll.push_back(position);
                // переходим к следующему индексу после position
                i = position + 1;
            }
	return positionAll;
	}

	//поиск адресов картинок
	std::vector< std::string > findUrl(std::string text) 
    {
        int ind=0;
		std::vector< int > positionAllHttps; //список всех позиций слова https
		std::string wordBegin {"https"};     // слово для поиска
		std::vector< int > positionAllJpg;   //список всех позиций слова jpg
		std::string wordEnd {".jpg"};        // слово для поиска
		std::vector< int > positionHttps;    //список синхронизированных позиций слова https
		std::vector< int > positionJpg;      //список синхронизированных позиций слова jpg
		std::vector< std::string > arrUrl;   // список найденных url

		positionAllHttps=findSubstring(text,wordBegin); // поиск всех слов https в тексте
		positionAllJpg=findSubstring(text,wordEnd);     // поиск всех слов jpg в тексте

		for (unsigned long i=0; i<positionAllHttps.size()-1;i++ )
		{
            if( positionAllJpg[ind]> positionAllHttps[i+1])
			{}
                else
                    if( positionAllJpg[ind]< positionAllHttps[i+1])
                        {
                            positionHttps.push_back(positionAllHttps[i]);
                            positionJpg.push_back(positionAllJpg[ind]);
                            ++ind;
                        }
		}

		for (unsigned long i=0; i<positionHttps.size()-1;i++ )
		{
			if (positionJpg[i]-positionHttps[i]+4>0)
			{
				int sizeBuffer=positionJpg[i]-positionHttps[i]+4;
				std::string strA = text.substr(positionHttps[i], sizeBuffer);
				strA = std::regex_replace(strA, std::regex("%3A%2F%2F"), "://"); 
				strA = std::regex_replace(strA, std::regex("%2F"), "/");
				//std::cout << strA << '\n';
				arrUrl.push_back(strA);
			}
		}
		return arrUrl;
	}

	int saveFile(std::string nameFile, std::vector< std::string > arrUrl)
	{
        std::ofstream output_file(nameFile);
        std::ostream_iterator<std::string> output_iterator(output_file, "\n");
            try 
                {
                    std::copy(std::begin(arrUrl), std::end(arrUrl), output_iterator);
                }
            catch (std::exception& e) 
                {
                    return -1;
                }
        return 0;
	}
	void downloadFoto(std::vector< std::string > arrUrl, std::string patch, std::string nameFile,int count )
	{
		for (unsigned long i=0; i<arrUrl.size()-1;i++ )
		{
            if (i>count)
                break;
            std::string nameShag = std::to_string(i);
            std::string str1 = "wget -nd -r -P ";
            std::string str3 = " -A jpeg,jpg,bmp,gif,png ";
            std::string concat_str = str1 +  str3 + " -O "+ patch+nameFile + nameShag +".jpg ";;
            //std::string cmd = concat_str + arrUrl[i]+ " -O "+ nameFile + nameShag +".jpg ";
            std::string cmd = concat_str + arrUrl[i];
            system(cmd.c_str());
		}
	}

	void deleleSmall(std::string filepath,int rows,int cols) 
	{
        std::string path = filepath;
			for (const auto & entry : std::filesystem::directory_iterator(path))
			{
                cv::Mat src = cv::imread( entry.path().string() );
			    if (src.rows <= rows && src.cols <= cols)
                    remove(entry.path());
			}
	}

	void copyFile(std::string  filepath,std::string  target, int percent) 
	{
		std::string path = filepath;
		std::string train =  "train";
		std::string test =  "test";
		std::string pathTrain=target;
		std::string pathTest=target;
		pathTrain+=train;
		pathTest+=test;
		std::string pathRez;
			for (const auto & entry : std::filesystem::directory_iterator(path))
			{
				std::filesystem::path sourceFile = entry.path();
					if ((rand()%100)<percent)
						pathRez=pathTrain;
					else
						pathRez=pathTest;
					std::filesystem::path targetParent = pathRez;
					auto target = targetParent / sourceFile.filename(); 
					try 
					{
						std::filesystem::create_directories(targetParent); 
						std::filesystem::copy_file(sourceFile, target, std::filesystem::copy_options::overwrite_existing);
					}
					catch (std::exception& e) 
					{
						std::cout << e.what();
					}
			}
	}
	
	
std::string	pythonDownloadYandex()
{
        FILE *pipe = popen("python3 /home/oleg/kodII/mrcv/python/url.py", "r"); // Replace 'your_script.py' with your Python script's filename

            if (!pipe) {
                std::cerr << "Failed to open pipe." << std::endl;
                return "0";
            }

            char buffer[128];
            std::string result = "";
                while (!feof(pipe)) {
                    if (fgets(buffer, 128, pipe) != NULL)
                        result += buffer;
                }
            pclose(pipe);

           // std::cout << "Python script output:\n" << result << std::endl;

    return result;
}
int getImagesFromYandex(std::string patchBody, int count, int minwidth, int minheight, std::string nametemplate, std::string outputfolder, bool separatedataset, int trainsetpercentage)
{
    
    //std::string text= readFile(patchBody); // читаем скачанный файл
   // std::string text=parserUrlString(patchBody.c_str());
    std::string text= pythonDownloadYandex();
    std::vector< std::string > arrUrl;
    arrUrl =findUrl(text); //поиск в файле url
   // int rez=saveFile("url.txt",arrUrl); // запись в файл список url
    downloadFoto(arrUrl,outputfolder,nametemplate,count); // качаем все фото из списка
    deleleSmall(outputfolder,minwidth,minheight); // удаляем мелкие фото (ширина, высота)
    if (separatedataset)
        copyFile(outputfolder,outputfolder,trainsetpercentage); // раскладываем файлы по папкам в процентном отношении 70% 
    
    return 0;
    
    
    
    
}


}
