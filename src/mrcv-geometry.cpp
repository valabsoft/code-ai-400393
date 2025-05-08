#include "mrcv/mrcv-geometry.h"

namespace mrcv
{
	std::vector<Cloud3DItem> geometryLoadData(std::string pathtofile, int cluster, int rows, int cols, bool norm)
	{
		std::vector<Cloud3DItem> cloud3D; // Облако 3D-точек

		if (std::filesystem::exists(pathtofile))
		{
			std::ifstream file(pathtofile); // Файловый поток

			if (file.is_open())
			{
				std::string line;	// Буферная строка
				std::string token;	// Текущее значение
				Cloud3DItem item;	// Текущаая точка облака

				int counter = 0;

				while (std::getline(file, line))
				{
					std::istringstream iss(line); // Строковый поток, связанный с текущей строкой файла

					// Начальная инициализация структуры
					item.U = 0;
					item.V = 0;
					item.X = 0;
					item.Y = 0;
					item.Z = 0;
					item.R = 0;
					item.G = 0;
					item.B = 0;
					item.C = 0;

					for (int i = 0; i < 9; i++)
					{
						iss >> token;
						std::stringstream ss(token);

						switch (i)
						{
						case 0:
							ss >> item.U;
							break;
						case 1:
							ss >> item.V;
							break;
						case 2:
							ss >> item.X;
							break;
						case 3:
							ss >> item.Y;
							break;
						case 4:
							ss >> item.Z;
							break;
						case 5:
							ss >> item.R;
							break;
						case 6:
							ss >> item.G;
							break;
						case 7:
							ss >> item.B;
							break;
						case 8:
							ss >> item.C;
							break;
						default:
							break;
						}
					}

					// Сохраняем данные только выбранного кластера
					if (item.C == cluster)
						cloud3D.push_back(item);
				}

				file.close();
			}
		}
		return cloud3D;
	}

	double geometryGetDistance(Point3D p1, Point3D p2)
	{
		return std::sqrt((p2.X - p1.X) * (p2.X - p1.X) +
			(p2.Y - p1.Y) * (p2.Y - p1.Y) +
			(p2.Z - p1.Z) * (p2.Z - p1.Z));
	}

	size_t geometryGetNumberOfNearestPoints(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, Point3D MN, Point3D M0)
	{
		// Пороговое значение расстояния
		double distanceThreshold = geometryGetDistance(MN, M0);
		size_t N = 0;
		for (size_t i = 0; i < X.size(); i++) {
			double dist = geometryGetDistance(MN, Point3D(X[i], Y[i], Z[i], "M"));
			if (dist < distanceThreshold / 2)
				N += 1;
		}
		return N;
	}

	double geometryLineSegDistance(std::vector<double> vertexP, std::vector<double> vertexP0, std::vector<double> vertexP1)
	{
		std::vector<double> subtractionP1P0 = geomentryVectorSubtraction(vertexP1, vertexP0);
		std::vector<double> subtractionPP0 = geomentryVectorSubtraction(vertexP, vertexP0);
		double area = geometryVectorNorm(geometryVectroCross(subtractionP1P0, subtractionPP0));
		double CD = area / geometryVectorNorm(subtractionP1P0);
		return CD;
	}

	std::vector<double> geomentryVectorSubtraction(std::vector<double> A, std::vector<double> B)
	{
		std::vector<double> result = A;
		for (size_t i = 0; i < A.size(); i++)
		{
			result[i] -= B[i];
		}
		return result;
	}

	double geometryVectorNorm(std::vector<double> A)
	{
		double res = 0;
		for (size_t i = 0; i < A.size(); i++)
		{
			res += std::pow(A[i], 2);
		}
		return std::sqrt(std::abs(res));
	}
	
	std::vector<double> geometryVectroCross(std::vector<double> A, std::vector<double> B)
	{
		double x1 = A[1] * B[2] - A[2] * B[1];
		double y1 = A[2] * B[0] - A[0] * B[2];
		double z1 = A[0] * B[1] - A[1] * B[0];
		
		return { x1, y1, z1 };
	}

	int geometryCalculateSize(std::vector<Cloud3DItem> cloud3D, double* l, double* w, double* h, double* length, double* width, double* distance)
	{
		// Векторы координат облака 3D-точек
		std::vector<double> X;
		std::vector<double> Y;
		std::vector<double> Z;

		// Формируем векторы из облака облака 3D-точек
		for (size_t i = 0; i < cloud3D.size(); i++)
		{
			X.push_back(cloud3D.at(i).X);
			Y.push_back(cloud3D.at(i).Y);
			Z.push_back(cloud3D.at(i).Z);
		}

		// Определение максимумов и минимумов для вычисления граничных точек
		double Xmin = *min_element(X.begin(), X.end());
		double Xmax = *max_element(X.begin(), X.end());

		double Ymin = *min_element(Y.begin(), Y.end());
		double Ymax = *max_element(Y.begin(), Y.end());

		double Zmin = *min_element(Z.begin(), Z.end());
		double Zmax = *max_element(Z.begin(), Z.end());

		// Центр масс
		double X0 = Xmin + (Xmax - Xmin) / 2.0;
		double Y0 = Ymin + (Ymax - Ymin) / 2.0;
		double Z0 = Zmin + (Zmax - Zmin) / 2.0;

		// Граничные точки
		Point3D M0 = Point3D(X0, Y0, Z0, "M0");
		Point3D M1 = Point3D(Xmin, Ymin, Zmin, "M1");
		Point3D M2 = Point3D(Xmax, Ymin, Zmin, "M2");
		Point3D M3 = Point3D(Xmax, Ymax, Zmin, "M3");
		Point3D M4 = Point3D(Xmin, Ymax, Zmin, "M4");
		Point3D M5 = Point3D(Xmin, Ymin, Zmax, "M5");
		Point3D M6 = Point3D(Xmax, Ymin, Zmax, "M6");
		Point3D M7 = Point3D(Xmax, Ymax, Zmax, "M7");
		Point3D M8 = Point3D(Xmin, Ymax, Zmax, "M8");

		// Поиск граничных точек, через которые пройдет ось
		size_t m1 = geometryGetNumberOfNearestPoints(X, Y, Z, M1, M0);
		size_t m2 = geometryGetNumberOfNearestPoints(X, Y, Z, M2, M0);
		size_t m3 = geometryGetNumberOfNearestPoints(X, Y, Z, M3, M0);
		size_t m4 = geometryGetNumberOfNearestPoints(X, Y, Z, M4, M0);
		size_t m5 = geometryGetNumberOfNearestPoints(X, Y, Z, M5, M0);
		size_t m6 = geometryGetNumberOfNearestPoints(X, Y, Z, M6, M0);
		size_t m7 = geometryGetNumberOfNearestPoints(X, Y, Z, M7, M0);
		size_t m8 = geometryGetNumberOfNearestPoints(X, Y, Z, M8, M0);

		// Устанавливаем кол-во точек около габаритной точки
		M1.setNumberOfPoint(m1);
		M2.setNumberOfPoint(m2);
		M3.setNumberOfPoint(m3);
		M4.setNumberOfPoint(m4);
		M5.setNumberOfPoint(m5);
		M6.setNumberOfPoint(m6);
		M7.setNumberOfPoint(m7);
		M8.setNumberOfPoint(m8);

		// Формируем список габаритных точек
		std::list<Point3D> M;
		M.push_back(M1);
		M.push_back(M2);
		M.push_back(M3);
		M.push_back(M4);
		M.push_back(M5);
		M.push_back(M6);
		M.push_back(M7);
		M.push_back(M8);

		// Сортировка списка габаритных точек
		M.sort();
		M.reverse();

		// Формируем список осевых точек
		std::list<Point3D> P;

		// Первая точка берется из отсортированного списка MD
		P.push_back(M.front());
		auto& P0 = M.front();

		// Вспомогательные переменные
		double distM0;
		double distP0;
		bool sameX;
		bool sameY;
		bool sameZ;
		bool sameXYZ;

		// Начинаем проверку со второй точки
		for (auto it = std::next(M.begin()); it != M.end(); ++it)
		{
			distM0 = geometryGetDistance(*it, M0);  // Расстояние от текущей точки до центра масс
			distP0 = geometryGetDistance(*it, P0);  // Расстояние между точками P0 и текущей

			// Проверяем, не лежат ли точки в одной плоскости
			sameX = it->X == P0.X;
			sameY = it->Y == P0.Y;
			sameZ = it->Z == P0.Z;
			sameXYZ = !(sameX || sameY || sameZ);			
			if ((distM0 < distP0) && sameXYZ)
			{
				// Добавляем найденную точку в список
				P.push_back(*it);
				// Если нужная точка найдена, прекращаем перебор
				break;
			}
		}
		auto& P1 = P.back();

		// Вычисление длины
		double objLength = sqrt(pow(P1.X - P0.X, 2) + pow(P1.Y - P0.Y, 2) + pow(P1.Z - P0.Z, 2));

		// Вычисление ширины
		std::vector<double> dists;
		for (size_t i = 0; i < X.size(); i++) {
			std::vector<double> vertexP0 = { P0.X, P0.Y, P0.Z };
			std::vector<double> vertexP1 = { P1.X, P1.Y, P1.Z };
			std::vector<double> points = { X[i], Y[i], Z[i] };
			dists.push_back(geometryLineSegDistance(points, vertexP0, vertexP1));
		}
		double objWidth = *max_element(dists.begin(), dists.end()) * 2;

		// Вывод габаритов параллелепипеда, где (L-длина W-ширина H-высота )
		*l = geometryGetDistance(M1, M2);
		*w = geometryGetDistance(M2, M3);
		*h = geometryGetDistance(M3, M7);

		*length = objLength;
		*width = objWidth;

		// Расчет расстояния до камеры
		Point3D CaM0 = Point3D(0, 0, 0, "CaM0");
		*distance = geometryGetDistance(CaM0, M0);

		return EXIT_SUCCESS;
	}
}