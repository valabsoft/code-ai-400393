#include <mrcv/mrcv.h>

#include <gtest/gtest.h>

TEST(add_test, add_3_2)
{
    EXPECT_EQ(mrcv::add(3, 2), 5);
	
 

}

TEST(add_test, save_File)
{
    std::string patch="/home/oleg/kodII/mrcv/body.html"; //место где лежит скачанный файл
    std::string text= mrcv::readFile(patch);
    std::vector< std::string > arrUrl;
    arrUrl =mrcv::urlFind(text);     
    
    EXPECT_EQ(mrcv::saveFile("url.txt",arrUrl),0);
}


