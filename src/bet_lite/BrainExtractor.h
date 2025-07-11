#ifndef BRAIN_EXTRACTOR_H
#define BRAIN_EXTRACTOR_H

#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ITK Core
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

// ITK Preprocessing
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

// ITK Thresholding and Segmentation
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"

// ITK Morphological Operations
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryFillholeImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"

// ITK Smoothing
#include "itkAntiAliasBinaryImageFilter.h"

// ITK Statistics
#include "itkImageToHistogramFilter.h"
#include "itkListSample.h"
#include "itkStatisticsImageFilter.h"

// Custom validation metrics
#include "ValidationMetrics.h"

// Processing status enumeration
enum class ExtractionStatus {
  Success,
  InputImageInvalid,
  BiasFieldCorrectionFailed,
  ThresholdingFailed,
  MorphologyFailed,
  NoValidBrainRegion,
  OutputWriteFailed
};

// Exception class for brain extraction errors
class BrainExtractionException : public std::exception {
private:
  ExtractionStatus m_status;
  std::string m_message;

  std::string CreateErrorMessage(ExtractionStatus status,
                                 const std::string &details);

public:
  BrainExtractionException(ExtractionStatus status,
                           const std::string &details = "");
  const char *what() const noexcept override;
  ExtractionStatus GetStatus() const { return m_status; }
};

// Parameters structure for brain extraction
struct BrainExtractionParams {
  // Preprocessing parameters
  bool enable_bias_correction = true;
  float gaussian_sigma = 1.0f; // Gaussian filter standard deviation

  // Thresholding parameters
  float intensity_percentile = 0.02f; // Background intensity percentile
  float brain_percentile = 0.85f;     // Brain tissue intensity percentile
  bool adaptive_thresholding = true;  // Use adaptive vs fixed threshold

  // Morphological parameters
  float opening_radius = 2.0f; // Opening operation radius (mm)
  float closing_radius = 3.0f; // Closing operation radius (mm)
  bool fill_holes = true;      // Fill internal holes

  // Post-processing parameters
  bool enable_smoothing = true;
  int smoothing_iterations = 100;
  float smoothing_tolerance = 0.1f;

  // Debug and output options
  bool save_intermediate_results = false;
  std::string debug_output_dir = "./debug/";
  bool verbose = false;
};

// Progress reporting callback
using ProgressCallback =
    std::function<void(float percentage, const std::string &stage)>;

// Note: ValidationMetrics struct moved to ValidationMetrics.h for comprehensive
// implementation

// Main brain extraction class
class BrainExtractor {
public:
  using PixelType = float;
  using ImageType = itk::Image<PixelType, 3>;
  using MaskType = itk::Image<unsigned char, 3>;
  using BallStructuringElementType =
      itk::BinaryBallStructuringElement<MaskType::PixelType, 3>;

  // Constructor and destructor
  BrainExtractor();
  ~BrainExtractor() = default;

  // Input/Output file management
  void SetInputFile(const std::string &input);
  void SetOutputFile(const std::string &output);

  // Parameter management
  void SetParameters(const BrainExtractionParams &params);
  BrainExtractionParams GetParameters() const { return m_params; }

  // Progress reporting
  void SetProgressCallback(ProgressCallback callback);

  // Main execution function
  ExtractionStatus Execute();

  // Validation (if ground truth is available)
  ValidationMetrics ValidateResult(const std::string &ground_truth_file);
  ComprehensiveValidationMetrics
  ValidateResultComprehensive(const std::string &ground_truth_file);

  // Get intermediate results for debugging
  ImageType::Pointer GetPreprocessedImage() const {
    return m_preprocessedImage;
  }
  MaskType::Pointer GetRawMask() const { return m_rawMask; }
  MaskType::Pointer GetFinalMask() const { return m_finalMask; }

private:
  // Member variables
  std::string m_inputFile;
  std::string m_outputFile;
  BrainExtractionParams m_params;
  ProgressCallback m_progressCallback;

  // Image pointers for different processing stages
  ImageType::Pointer m_inputImage;
  ImageType::Pointer m_preprocessedImage;
  MaskType::Pointer m_rawMask;
  MaskType::Pointer m_finalMask;

  // Internal processing functions
  ExtractionStatus ReadInputImage();
  ExtractionStatus PreprocessImage();
  ExtractionStatus GenerateInitialMask();
  ExtractionStatus RefineMask();
  ExtractionStatus SaveResult();

  // Specific preprocessing steps
  ExtractionStatus ApplyBiasFieldCorrection();
  ExtractionStatus ApplyNoiseReduction();
  ExtractionStatus NormalizeIntensity();

  // Thresholding methods
  ExtractionStatus ApplyAdaptiveThresholding();
  float CalculateOptimalThreshold();

  // Morphological operations
  ExtractionStatus ApplyMorphologicalOperations();
  ExtractionStatus ExtractLargestComponent();
  ExtractionStatus FillHoles();
  ExtractionStatus SmoothBoundaries();

  // Utility functions
  void ReportProgress(float percentage, const std::string &stage);
  void SaveIntermediateResult(ImageType::Pointer image,
                              const std::string &suffix);
  void SaveIntermediateResult(MaskType::Pointer mask,
                              const std::string &suffix);
  BallStructuringElementType CreateStructuringElement(float radius_mm);

  // Validation helpers
  ValidationMetrics CalculateValidationMetrics(MaskType::Pointer result,
                                               MaskType::Pointer reference);
  ComprehensiveValidationMetrics
  CalculateComprehensiveMetrics(MaskType::Pointer result,
                                MaskType::Pointer reference);
  // Individual metric calculations
  OverlapMetrics CalculateOverlapMetrics(MaskType::Pointer result,
                                         MaskType::Pointer reference);
  BoundaryMetrics CalculateBoundaryMetrics(MaskType::Pointer result,
                                           MaskType::Pointer reference);
  VolumetricMetrics CalculateVolumetricMetrics(MaskType::Pointer result,
                                               MaskType::Pointer reference);
  MorphologyMetrics CalculateMorphologyMetrics(MaskType::Pointer mask);
  RobustnessMetrics CalculateRobustnessMetrics(MaskType::Pointer result,
                                               MaskType::Pointer reference);
  ClinicalMetrics
  CalculateClinicalMetrics(const ComprehensiveValidationMetrics &metrics);
};

#endif // BRAIN_EXTRACTOR_H