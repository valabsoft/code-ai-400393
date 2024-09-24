#include <mrcv/mrcv.h>

int main() {

    mrcv::DenseStereo denseStereo;
    std::string filename = "C:/Users/delis/Desktop/pointsClaster_1001.txt";
    denseStereo.loadDataFromFile(filename);
    denseStereo.makeClustering();
    denseStereo.printClusters();

    return 0;
}