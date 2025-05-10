#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(add_test, add_3_2)
{
    EXPECT_EQ(mrcv::add(3, 2), 5);
}