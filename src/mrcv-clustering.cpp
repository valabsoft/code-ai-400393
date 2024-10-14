#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>
#include <fstream>  // Добавлено для записи в файл

namespace mrcv
{
    // Загружает данные из указанного файла
    void DenseStereo::loadDataFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Не удалось открыть файл: " << filename << std::endl;
            return;
        }

        std::string line;
        // Очищаем предыдущие данные
        vuxyzrgb.xyz.clear();
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> point(3);
            double temp;

            iss >> temp >> temp; // Пропускаем ненужные данные
            iss >> point[0] >> point[1] >> point[2]; // Считываем координаты
            iss >> temp >> temp >> temp; // Пропускаем ненужные данные
            iss >> temp; // Пропускаем ненужные данные

            vuxyzrgb.xyz.push_back(point); // Добавляем точку в вектор
        }

        file.close();
    }

    // Вычисляет расстояния между точками в 3D-пространстве
    void computeDistances(size_t start, size_t end, const std::vector<double>& x3D,
        const std::vector<double>& y3D, const std::vector<double>& z3D,
        std::vector<std::vector<double>>& S, double& sumS) {
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < x3D.size(); ++j) {
                S[i][j] = std::sqrt(std::pow((x3D[i] - x3D[j]), 2) +
                    std::pow((y3D[i] - y3D[j]), 2) +
                    std::pow((z3D[i] - z3D[j]), 2));
                sumS += S[i][j];
            }
        }
    }

    // Нормализует расстояния
    void normalizeDistances(size_t start, size_t end, std::vector<std::vector<double>>& S,
        std::vector<std::vector<double>>& S3, double S2) {
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < S[i].size(); ++j) {
                S3[i][j] = S[i][j] / S2;
            }
        }
    }

    // Вычисляет близость точек
    void computeProximity(size_t start, size_t end, const std::vector<std::vector<double>>& S3,
        std::vector<std::vector<bool>>& L, double coef) {
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < S3[i].size(); ++j) {
                if (S3[i][j] < coef)
                    L[i][j] = true;
            }
        }
    }

    // Кластеризация загруженных данных
    void DenseStereo::makeClustering() {
        if (vuxyzrgb.xyz.empty()) return;

        // =========== Mutex
        vuxyzrgb_mutex.lock();
        std::vector<std::vector<double>> xyz = vuxyzrgb.xyz;
        vuxyzrgb_mutex.unlock();

        // Инициализация
        float coef = 0.083;
        std::vector<double> x3D, y3D, z3D;
        size_t Len3D = xyz.size();
        for (size_t i = 0; i < Len3D; i++) {
            x3D.push_back(xyz[i][0]);
            y3D.push_back(xyz[i][1]);
            z3D.push_back(xyz[i][2]);
        }

        // Создание Матрицы расстояний
        std::vector<std::vector<double>> S(Len3D, std::vector<double>(Len3D, 0));
        double sumOfS = 0.0;
        size_t numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<double> partialSums(numThreads, 0.0);

        size_t blockSize = Len3D / numThreads;
        for (size_t t = 0; t < numThreads; ++t) {
            size_t start = t * blockSize;
            size_t end = (t == numThreads - 1) ? Len3D : start + blockSize;
            threads.emplace_back(computeDistances, start, end, std::cref(x3D), std::cref(y3D), std::cref(z3D), std::ref(S), std::ref(partialSums[t]));
        }

        for (auto& t : threads) {
            t.join();
        }

        for (const auto& ps : partialSums) {
            sumOfS += ps;
        }

        double S2 = sumOfS / double(Len3D * Len3D);
        std::vector<std::vector<double>> S3(Len3D, std::vector<double>(Len3D, 0.0));
        threads.clear();

        for (size_t t = 0; t < numThreads; ++t) {
            size_t start = t * blockSize;
            size_t end = (t == numThreads - 1) ? Len3D : start + blockSize;
            threads.emplace_back(normalizeDistances, start, end, std::ref(S), std::ref(S3), S2);
        }

        for (auto& t : threads) {
            t.join();
        }

        std::vector<std::vector<bool>> L(Len3D, std::vector<bool>(Len3D, false));
        threads.clear();

        for (size_t t = 0; t < numThreads; ++t) {
            size_t start = t * blockSize;
            size_t end = (t == numThreads - 1) ? Len3D : start + blockSize;
            threads.emplace_back(computeProximity, start, end, std::cref(S3), std::ref(L), coef);
        }

        for (auto& t : threads) {
            t.join();
        }

        IDX.resize(Len3D, -1);
        std::vector<std::vector<std::vector<double>>> clastersData;
        int numClaster = 0, ID_current = 0;
        std::vector<int> IDs_check, IDs_clusters, IDs_lost;
        IDs_check.push_back(0);

        for (int i = 0; i < int(Len3D); i++) {
            if (L[0][i])
                IDs_clusters.push_back(i);
            else
                IDs_lost.push_back(i);
        }

        while (!IDs_lost.empty()) {
            ID_current = IDs_lost[0];
            IDs_clusters.clear();

            for (int i = 0; i < int(Len3D); i++)
                if (L[ID_current][i])
                    IDs_clusters.push_back(i);

            std::vector<int> VectDiff;
            for (int i = 0; i < int(IDs_clusters.size()); i++) {
                auto it = std::find(IDs_check.begin(), IDs_check.end(), IDs_clusters[i]);
                if (it == IDs_check.end())
                    VectDiff.push_back(IDs_clusters[i]);
            }

            while (!VectDiff.empty()) {
                ID_current = VectDiff[0];

                for (int i = 0; i < int(Len3D); i++)
                    if (L[ID_current][i]) {
                        auto it = std::find(IDs_clusters.begin(), IDs_clusters.end(), i);
                        if (it == IDs_clusters.end())
                            IDs_clusters.push_back(i);
                    }

                auto it1 = std::find(IDs_check.begin(), IDs_check.end(), ID_current);
                if (it1 == IDs_check.end())
                    IDs_check.push_back(ID_current);

                VectDiff.clear();
                for (int i = 0; i < int(IDs_clusters.size()); i++) {
                    auto it = std::find(IDs_check.begin(), IDs_check.end(), IDs_clusters[i]);
                    if (it == IDs_check.end())
                        VectDiff.push_back(IDs_clusters[i]);
                }
            }

            VectDiff.clear();
            for (int i = 0; i < int(IDs_lost.size()); i++) {
                auto it = std::find(IDs_clusters.begin(), IDs_clusters.end(), IDs_lost[i]);
                if (it == IDs_clusters.end())
                    VectDiff.push_back(IDs_lost[i]);
            }
            IDs_lost.clear();
            IDs_lost = VectDiff;

            std::vector<std::vector<double>> xyz2;

            for (int i = 0; i < int(IDs_clusters.size()); i++) {
                xyz2.push_back({ x3D[IDs_clusters[i]], y3D[IDs_clusters[i]], z3D[IDs_clusters[i]] });
                IDX[IDs_clusters[i]] = numClaster;
            }
            numClaster++;
            clastersData.push_back(xyz2);
        }

        // Запись кластеров и точек в файл
        std::ofstream outputFile("files/clusters_data.txt");
        if (outputFile.is_open()) {
            outputFile << "Координаты точки, номер точки, номер кластера" << std::endl;
            for (size_t i = 0; i < Len3D; ++i) {
                outputFile << x3D[i] << ", " << y3D[i] << ", " << z3D[i]
                    << ", " << i << ", " << IDX[i] << std::endl;
            }
            outputFile.close();
        }
    }

    // Логирование успешной кластеризации
    void DenseStereo::printClusters() {
        std::ostringstream logStream;  // Строковый поток для формирования сообщения лога
        logStream << "Кластеризация выполнена успешно. Количество точек: " << IDX.size() << ", количество кластеров: " << *std::max_element(IDX.begin(), IDX.end()) + 1;
        // Записываем сообщение в лог
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
    }

}
