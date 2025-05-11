#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(sensorsfusion_test, sensorsfusion)
{
	auto currentPath = std::filesystem::current_path();
	std::filesystem::path path = currentPath / "data" / "sensorsfusion";

	auto datasetPath = path / "images";
	auto imuDataPath = path / "imuLog.csv";
	auto usblDataPath = path / "usblLog.csv";
	auto fuseTupleSavePath = path / "fuseData.yaml";

	int exitcode = mrcv::fuseSensorData(usblDataPath.u8string(), imuDataPath.u8string(), datasetPath.u8string(), fuseTupleSavePath.u8string(), false);

	EXPECT_EQ(exitcode, EXIT_SUCCESS);
}