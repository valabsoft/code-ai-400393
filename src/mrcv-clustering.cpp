#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

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
		while (getline(file, line)) {
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

		vuxyzrgb_mutex.lock();
		std::vector<std::vector<double>> xyz = vuxyzrgb.xyz;
		vuxyzrgb_mutex.unlock();

		float proximityCoefficient = 0.083;
		std::vector<double> x3D, y3D, z3D;

		size_t len3D = xyz.size();
		for (size_t i = 0; i < len3D; i++) {
			x3D.push_back(xyz[i][0]);
			y3D.push_back(xyz[i][1]);
			z3D.push_back(xyz[i][2]);
		}

		std::vector<std::vector<double>> distanceMatrix(len3D, std::vector<double>(len3D, 0));
		double sumOfDistances = 0.0;
		size_t numThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		std::vector<double> partialSums(numThreads, 0.0);

		size_t blockSize = len3D / numThreads;
		for (size_t t = 0; t < numThreads; ++t) {
			size_t start = t * blockSize;
			size_t end = (t == numThreads - 1) ? len3D : start + blockSize;
			threads.emplace_back(computeDistances, start, end, std::cref(x3D), std::cref(y3D),
				std::cref(z3D), std::ref(distanceMatrix), std::ref(partialSums[t]));
		}

		for (auto& t : threads) {
			t.join();
		}

		for (const auto& ps : partialSums) {
			sumOfDistances += ps;
		}

		double avgDistance = sumOfDistances / double(len3D * len3D);
		std::vector<std::vector<double>> normalizedDistances(len3D, std::vector<double>(len3D, 0.0));
		threads.clear();

		for (size_t t = 0; t < numThreads; ++t) {
			size_t start = t * blockSize;
			size_t end = (t == numThreads - 1) ? len3D : start + blockSize;
			threads.emplace_back(normalizeDistances, start, end, std::ref(distanceMatrix),
				std::ref(normalizedDistances), avgDistance);
		}

		for (auto& t : threads) {
			t.join();
		}

		std::vector<std::vector<bool>> proximityMatrix(len3D, std::vector<bool>(len3D, false));
		threads.clear();

		for (size_t t = 0; t < numThreads; ++t) {
			size_t start = t * blockSize;
			size_t end = (t == numThreads - 1) ? len3D : start + blockSize;
			threads.emplace_back(computeProximity, start, end, std::cref(normalizedDistances),
				std::ref(proximityMatrix), proximityCoefficient);
		}

		for (auto& t : threads) {
			t.join();
		}

		IDX.resize(len3D, -1);
		std::vector<std::vector<std::vector<double>>> clustersData;
		int clusterId = 0, currentId = 0;
		std::vector<int> checkedIds, clusterIds, remainingIds;
		checkedIds.push_back(0);

		for (int i = 0; i < int(len3D); i++) {
			if (proximityMatrix[0][i])
				clusterIds.push_back(i);
			else
				remainingIds.push_back(i);
		}

		while (!remainingIds.empty()) {
			currentId = remainingIds[0];
			clusterIds.clear();

			for (int i = 0; i < int(len3D); i++)
				if (proximityMatrix[currentId][i])
					clusterIds.push_back(i);

			std::vector<int> idDifference;
			for (int i = 0; i < int(clusterIds.size()); i++) {
				auto it = std::find(checkedIds.begin(), checkedIds.end(), clusterIds[i]);
				if (it == checkedIds.end())
					idDifference.push_back(clusterIds[i]);
			}

			while (!idDifference.empty()) {
				currentId = idDifference[0];

				for (int i = 0; i < int(len3D); i++)
					if (proximityMatrix[currentId][i]) {
						auto it = std::find(clusterIds.begin(), clusterIds.end(), i);
						if (it == clusterIds.end())
							clusterIds.push_back(i);
					}

				auto checkedIt = std::find(checkedIds.begin(), checkedIds.end(), currentId);
				if (checkedIt == checkedIds.end())
					checkedIds.push_back(currentId);

				idDifference.clear();
				for (int i = 0; i < int(clusterIds.size()); i++) {
					auto it = std::find(checkedIds.begin(), checkedIds.end(), clusterIds[i]);
					if (it == checkedIds.end())
						idDifference.push_back(clusterIds[i]);
				}
			}

			idDifference.clear();
			for (int i = 0; i < int(remainingIds.size()); i++) {
				auto it = std::find(clusterIds.begin(), clusterIds.end(), remainingIds[i]);
				if (it == clusterIds.end())
					idDifference.push_back(remainingIds[i]);
			}

			for (auto it = remainingIds.begin(); it != remainingIds.end();) {
				if (std::find(idDifference.begin(), idDifference.end(), *it) != idDifference.end())
					it = remainingIds.erase(it);
				else
					++it;
			}

			std::vector<std::vector<double>> newClusterData;
			for (int i = 0; i < int(clusterIds.size()); i++) {
				IDX[clusterIds[i]] = clusterId;
				newClusterData.push_back({ x3D[clusterIds[i]], y3D[clusterIds[i]], z3D[clusterIds[i]] });
			}

			clustersData.push_back(newClusterData);
			clusterId++;
		}

		std::cout << "Количество кластеров: " << *std::max_element(IDX.begin(), IDX.end()) + 1 << std::endl;
		for (int i = 0; i <= *std::max_element(IDX.begin(), IDX.end()); ++i) {
			std::cout << "Кластер " << i << ":\n";
			for (size_t j = 0; j < IDX.size(); ++j) {
				if (IDX[j] == i) {
					std::cout << "Точка " << j << ": (" << vuxyzrgb.xyz[j][0] << ", "
						<< vuxyzrgb.xyz[j][1] << ", " << vuxyzrgb.xyz[j][2] << ")\n";
				}
			}
		}
	}
}
