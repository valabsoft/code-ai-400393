#include <mrcv/mrcv.h>

int main() {
 
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    
    auto dataPath = path / "claster.dat";

    mrcv::DenseStereo denseStereo;    
    denseStereo.loadDataFromFile(dataPath.u8string());
    denseStereo.makeClustering();
    denseStereo.printClusters();

    return 0;
}
