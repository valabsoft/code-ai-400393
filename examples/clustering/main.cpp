#include <mrcv/mrcv-clustering.h>

int main() {

    mrcv::DenseStereo denseStereo;
    std::string filename = "pointsClaster_1001.txt";
    denseStereo.loadDataFromFile(filename);
    denseStereo.Clustering();
    denseStereo.printClusters();
    denseStereo.visualizeClusters3D();

    return 0;
}