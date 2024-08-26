#include <mrcv/mrcv.h>

int main() {

    mrcv::DenseStereo denseStereo;
    std::string filename = "pointsClaster_1001.txt";
    denseStereo.loadDataFromFile(filename);
    denseStereo.Clustering();
    denseStereo.printClusters();

    return 0;
}