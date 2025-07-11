#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "../src/io/ImageIO.h"
#include "../src/io/CompatUtils.h"

using namespace neurocompass::io;

class ImageIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir = std::filesystem::temp_directory_path() / "neurocompass_io_test";
        std::filesystem::create_directories(test_dir);
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    std::filesystem::path test_dir;
};

TEST_F(ImageIOTest, Image3DBasicOperations) {
    // Test basic Image3D operations
    Image3D<float> image(64, 64, 32);
    
    EXPECT_TRUE(image.IsValid());
    EXPECT_EQ(image.GetSizeX(), 64);
    EXPECT_EQ(image.GetSizeY(), 64);
    EXPECT_EQ(image.GetSizeZ(), 32);
    EXPECT_EQ(image.GetTotalPixels(), 64 * 64 * 32);
    
    // Test filling
    image.Fill(42.0f);
    EXPECT_EQ(image(0, 0, 0), 42.0f);
    EXPECT_EQ(image(10, 20, 15), 42.0f);
    
    // Test individual pixel access
    image(10, 20, 15) = 100.0f;
    EXPECT_EQ(image(10, 20, 15), 100.0f);
    
    // Test statistics
    image.Fill(10.0f);
    EXPECT_FLOAT_EQ(image.GetMean(), 10.0f);
    EXPECT_FLOAT_EQ(image.GetMinValue(), 10.0f);
    EXPECT_FLOAT_EQ(image.GetMaxValue(), 10.0f);
    EXPECT_FLOAT_EQ(image.GetStandardDeviation(), 0.0f);
}

TEST_F(ImageIOTest, Image3DIndexOperations) {
    Image3D<int16_t> image(10, 10, 10);
    
    // Test index validation
    Image3D<int16_t>::IndexType valid_index = {{5, 5, 5}};
    Image3D<int16_t>::IndexType invalid_index = {{15, 5, 5}};
    
    EXPECT_TRUE(image.IsIndexValid(valid_index));
    EXPECT_FALSE(image.IsIndexValid(invalid_index));
    
    // Test linear/index conversion
    Image3D<int16_t>::IndexType index = {{3, 4, 5}};
    size_t linear = image.IndexToLinear(index);
    auto converted_back = image.LinearToIndex(linear);
    
    EXPECT_EQ(index[0], converted_back[0]);
    EXPECT_EQ(index[1], converted_back[1]);
    EXPECT_EQ(index[2], converted_back[2]);
}

TEST_F(ImageIOTest, CreateTestImage) {
    // Test utility function for creating test images
    auto image = ImageUtils::CreateTestImage<float>(32, 32, 16, 0.1);
    
    EXPECT_NE(image, nullptr);
    EXPECT_TRUE(image->IsValid());
    EXPECT_EQ(image->GetSizeX(), 32);
    EXPECT_EQ(image->GetSizeY(), 32);
    EXPECT_EQ(image->GetSizeZ(), 16);
    
    // Check that image has some variation (not all zeros)
    auto minmax = image->GetMinMax();
    EXPECT_LT(minmax.first, minmax.second);
}

TEST_F(ImageIOTest, NiftiWriteAndRead) {
    // Create test image
    auto original_image = ImageUtils::CreateTestImage<float>(32, 32, 16);
    ASSERT_NE(original_image, nullptr);
    
    // Set some metadata
    auto info = original_image->GetImageInfo();
    info.voxel_size = {{2.0, 2.0, 3.0}};
    info.origin = {{-16.0, -16.0, -24.0}};
    info.description = "Test image for NeuroCompass";
    original_image->SetImageInfo(info);
    
    // Write to NIfTI file
    std::string filename = (test_dir / "test_image.nii").string();
    
    ImageWriter writer;
    ASSERT_TRUE(writer.Open(filename, ImageFormat::NIFTI_1));
    
    ImageWriter::WriteOptions write_options;
    write_options.description = "Test NIfTI image";
    write_options.verbose = true;
    
    EXPECT_TRUE(writer.WriteImage(*original_image, write_options));
    
    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(filename));
    EXPECT_GT(std::filesystem::file_size(filename), 0);
    
    // Read back the image
    ImageReader reader;
    ASSERT_TRUE(reader.Open(filename));
    
    auto loaded_image = reader.ReadImageFloat();
    ASSERT_NE(loaded_image, nullptr);
    
    // Verify dimensions match
    EXPECT_EQ(loaded_image->GetSizeX(), original_image->GetSizeX());
    EXPECT_EQ(loaded_image->GetSizeY(), original_image->GetSizeY());
    EXPECT_EQ(loaded_image->GetSizeZ(), original_image->GetSizeZ());
    
    // Verify voxel size
    auto loaded_spacing = loaded_image->GetSpacing();
    EXPECT_FLOAT_EQ(loaded_spacing[0], 2.0);
    EXPECT_FLOAT_EQ(loaded_spacing[1], 2.0);
    EXPECT_FLOAT_EQ(loaded_spacing[2], 3.0);
    
    // Verify some pixel values (should be identical for float data)
    EXPECT_FLOAT_EQ((*loaded_image)(0, 0, 0), (*original_image)(0, 0, 0));
    EXPECT_FLOAT_EQ((*loaded_image)(10, 15, 8), (*original_image)(10, 15, 8));
    EXPECT_FLOAT_EQ((*loaded_image)(31, 31, 15), (*original_image)(31, 31, 15));
}

TEST_F(ImageIOTest, CompressedNiftiIO) {
    // Create test image
    auto image = ImageUtils::CreateTestImage<int16_t>(24, 24, 12);
    ASSERT_NE(image, nullptr);
    
    // Write compressed NIfTI file
    std::string filename = (test_dir / "test_compressed.nii.gz").string();
    
    ImageWriter::WriteOptions options;
    options.compress = true;
    options.verbose = true;
    
    EXPECT_TRUE(ImageUtils::WriteImage(*image, filename));
    
    // Verify compressed file was created
    EXPECT_TRUE(std::filesystem::exists(filename));
    
    // Read back the compressed image
    auto loaded_image = ImageUtils::ReadImage<int16_t>(filename);
    ASSERT_NE(loaded_image, nullptr);
    
    // Verify dimensions
    EXPECT_EQ(loaded_image->GetSize()[0], image->GetSize()[0]);
    EXPECT_EQ(loaded_image->GetSize()[1], image->GetSize()[1]);
    EXPECT_EQ(loaded_image->GetSize()[2], image->GetSize()[2]);
    
    // Compare some pixel values
    EXPECT_EQ((*loaded_image)(0, 0, 0), (*image)(0, 0, 0));
    EXPECT_EQ((*loaded_image)(12, 12, 6), (*image)(12, 12, 6));
}

TEST_F(ImageIOTest, TypeConversion) {
    // Create float image
    auto float_image = ImageUtils::CreateTestImage<float>(16, 16, 8);
    ASSERT_NE(float_image, nullptr);
    
    // Fill with known values
    float_image->Fill(123.456f);
    (*float_image)(8, 8, 4) = 1000.7f;
    
    // Convert to int16
    auto int16_image = ImageUtils::ConvertImageType<float, int16_t>(*float_image);
    ASSERT_NE(int16_image, nullptr);
    
    // Check conversion
    EXPECT_EQ((*int16_image)(0, 0, 0), 123);  // Truncated
    EXPECT_EQ((*int16_image)(8, 8, 4), 1000); // Truncated
    
    // Verify dimensions preserved
    EXPECT_EQ(int16_image->GetSize()[0], float_image->GetSize()[0]);
    EXPECT_EQ(int16_image->GetSize()[1], float_image->GetSize()[1]);
    EXPECT_EQ(int16_image->GetSize()[2], float_image->GetSize()[2]);
}

TEST_F(ImageIOTest, ImageInfo) {
    // Create test file
    std::string filename = (test_dir / "info_test.nii").string();
    EXPECT_TRUE(ImageUtils::SaveTestImage<float>(filename, 20, 30, 10));
    
    // Get image info without loading full data
    auto info = ImageUtils::GetImageInfo(filename);
    
    EXPECT_EQ(info.dimensions[0], 20);
    EXPECT_EQ(info.dimensions[1], 30);
    EXPECT_EQ(info.dimensions[2], 10);
    EXPECT_EQ(info.datatype, DataType::FLOAT32);
    EXPECT_GT(info.size_bytes, 0);
    
    // Test memory estimation
    size_t estimated_size = ImageUtils::EstimateMemoryUsage(filename);
    EXPECT_EQ(estimated_size, 20 * 30 * 10 * sizeof(float));
    
    // Test memory check
    EXPECT_TRUE(ImageUtils::CanLoadInMemory(filename, 100));  // 100MB should be enough
    EXPECT_FALSE(ImageUtils::CanLoadInMemory(filename, 0));   // 0MB definitely not enough
}

TEST_F(ImageIOTest, RegionOperations) {
    // Create larger image
    Image3D<uint8_t> image(20, 20, 10);
    image.Fill(100);
    
    // Set some specific values
    for (int i = 5; i < 15; ++i) {
        for (int j = 5; j < 15; ++j) {
            for (int k = 2; k < 8; ++k) {
                image(i, j, k) = 200;
            }
        }
    }
    
    // Extract a region
    Image3D<uint8_t>::IndexType start = {{5, 5, 2}};
    Image3D<uint8_t>::SizeType region_size = {{10, 10, 6}};
    
    auto region = image.ExtractRegion(start, region_size);
    
    EXPECT_EQ(region.GetSize()[0], 10);
    EXPECT_EQ(region.GetSize()[1], 10);
    EXPECT_EQ(region.GetSize()[2], 6);
    
    // Check that extracted region has correct values
    EXPECT_EQ(region(0, 0, 0), 200);
    EXPECT_EQ(region(5, 5, 3), 200);
    
    // Test setting a region back
    Image3D<uint8_t> target(30, 30, 15);
    target.Fill(50);
    
    Image3D<uint8_t>::IndexType target_start = {{10, 10, 5}};
    target.SetRegion(target_start, region);
    
    // Check that target now contains the region data
    EXPECT_EQ(target(10, 10, 5), 200);  // Start of copied region
    EXPECT_EQ(target(19, 19, 10), 200); // End of copied region
    EXPECT_EQ(target(0, 0, 0), 50);     // Outside region should be unchanged
}

TEST_F(ImageIOTest, ErrorHandling) {
    // Test opening non-existent file
    ImageReader reader;
    EXPECT_FALSE(reader.Open("non_existent_file.nii"));
    
    // Test reading invalid file
    std::string invalid_file = (test_dir / "invalid.nii").string();
    std::ofstream invalid(invalid_file);
    invalid << "This is not a valid NIfTI file";
    invalid.close();
    
    EXPECT_FALSE(reader.Open(invalid_file));
    
    // Test writing to invalid location (if possible to test)
    Image3D<float> image(10, 10, 10);
    std::string invalid_path = "/root/definitely_cannot_write_here.nii";
    
    // This might succeed or fail depending on permissions, so we don't assert
    ImageUtils::WriteImage(image, invalid_path);
}

TEST_F(ImageIOTest, FileUtilities) {
    // Test file utility functions
    std::string test_file = (test_dir / "test_utils.nii.gz").string();
    
    // Create a small test file
    std::ofstream file(test_file);
    file << "test content";
    file.close();
    
    // Test utilities
    EXPECT_TRUE(ImageUtils::FileExists(test_file));
    EXPECT_FALSE(ImageUtils::FileExists("definitely_does_not_exist.nii"));
    
    EXPECT_EQ(ImageUtils::GetFileExtension(test_file), ".nii.gz");
    EXPECT_EQ(ImageUtils::GetFileExtension("test.nii"), ".nii");
    EXPECT_EQ(ImageUtils::GetFileExtension("test.hdr"), ".hdr");
    
    std::string without_ext = ImageUtils::RemoveExtension(test_file);
    EXPECT_TRUE(compat::ends_with(without_ext, "test_utils"));
    
    EXPECT_GT(ImageUtils::GetFileSize(test_file), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}