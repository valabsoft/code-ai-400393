#include <mrcv/mrcv.h>

int main() {

    std::filesystem::path dataFile("files\\claster.dat");    
    auto currentPath = std::filesystem::current_path();
    auto dataPath = currentPath / dataFile;

    mrcv::DenseStereo denseStereo;    
    denseStereo.loadDataFromFile(dataPath.u8string());
    denseStereo.makeClustering();
    denseStereo.printClusters();

    return 0;
}