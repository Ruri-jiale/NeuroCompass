#include "AffineTransform.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

// ITK Headers
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

// Constructor
AffineTransform::AffineTransform(DegreesOfFreedom dof) : m_dof(dof) {
  m_transform = ITKTransformType::New();
  m_transform->SetIdentity();

  // Initialize parameters based on DOF
  int num_params = GetNumberOfParameters();
  m_parameters.resize(num_params, 0.0);
  UpdateParametersFromTransform();
}

// Copy constructor
AffineTransform::AffineTransform(const AffineTransform &other)
    : m_dof(other.m_dof), m_parameters(other.m_parameters) {
  m_transform = ITKTransformType::New();
  m_transform->SetParameters(other.m_transform->GetParameters());
  m_transform->SetFixedParameters(other.m_transform->GetFixedParameters());
}

// Assignment operator
AffineTransform &AffineTransform::operator=(const AffineTransform &other) {
  if (this != &other) {
    m_dof = other.m_dof;
    m_parameters = other.m_parameters;
    m_transform->SetParameters(other.m_transform->GetParameters());
    m_transform->SetFixedParameters(other.m_transform->GetFixedParameters());
  }
  return *this;
}

// Parameter management
void AffineTransform::SetParameters(const ParametersType &params) {
  if (!ValidateParameters(params, m_dof)) {
    throw std::invalid_argument(
        "Invalid parameters for specified degrees of freedom");
  }

  m_parameters = params;
  UpdateTransformFromParameters();
}

AffineTransform::ParametersType AffineTransform::GetParameters() const {
  return m_parameters;
}

int AffineTransform::GetNumberOfParameters() const {
  return static_cast<int>(m_dof);
}

// Degrees of freedom management
void AffineTransform::SetDegreesOfFreedom(DegreesOfFreedom dof) {
  m_dof = dof;
  int num_params = GetNumberOfParameters();
  m_parameters.resize(num_params);

  // Convert current transform to new parameterization
  UpdateParametersFromTransform();
}

// Identity and center management
void AffineTransform::SetIdentity() {
  m_transform->SetIdentity();
  std::fill(m_parameters.begin(), m_parameters.end(), 0.0);
}

void AffineTransform::SetCenter(const PointType &center) {
  m_transform->SetCenter(center);
}

AffineTransform::PointType AffineTransform::GetCenter() const {
  return m_transform->GetCenter();
}

// Euler angles and transformation setup
void AffineTransform::SetFromEulerAngles(double rx, double ry, double rz,
                                         double tx, double ty, double tz,
                                         double sx, double sy, double sz) {
  // Create rotation matrix from Euler angles (ZYX convention)
  MatrixType rotation =
      CreateRotationMatrix(DegToRad(rx), DegToRad(ry), DegToRad(rz));

  // Apply scaling
  MatrixType scaling = CreateScalingMatrix(sx, sy, sz);
  MatrixType matrix = rotation * scaling;

  // Set translation
  VectorType translation;
  translation[0] = tx;
  translation[1] = ty;
  translation[2] = tz;

  // Update ITK transform
  m_transform->SetMatrix(matrix);
  m_transform->SetTranslation(translation);

  // Update parameters
  UpdateParametersFromTransform();
}

// Get Euler angles decomposition
void AffineTransform::GetEulerAngles(double &rx, double &ry, double &rz) const {
  MatrixType matrix = m_transform->GetMatrix();

  // Remove scaling to get pure rotation
  VectorType scaling = ExtractScaling(matrix);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrix[i][j] /= scaling[i];
    }
  }

  DecomposeRotationMatrix(matrix, rx, ry, rz);
  rx = RadToDeg(rx);
  ry = RadToDeg(ry);
  rz = RadToDeg(rz);
}

AffineTransform::VectorType AffineTransform::GetTranslation() const {
  return m_transform->GetTranslation();
}

AffineTransform::VectorType AffineTransform::GetScaling() const {
  MatrixType matrix = m_transform->GetMatrix();
  return ExtractScaling(matrix);
}

// Matrix operations
void AffineTransform::SetMatrix(const MatrixType &matrix) {
  m_transform->SetMatrix(matrix);
  UpdateParametersFromTransform();
}

AffineTransform::MatrixType AffineTransform::GetMatrix() const {
  return m_transform->GetMatrix();
}

void AffineTransform::SetTranslation(const VectorType &translation) {
  m_transform->SetTranslation(translation);
  UpdateParametersFromTransform();
}

// Point and vector transformation
AffineTransform::PointType
AffineTransform::TransformPoint(const PointType &point) const {
  return m_transform->TransformPoint(point);
}

AffineTransform::VectorType
AffineTransform::TransformVector(const VectorType &vector) const {
  return m_transform->TransformVector(vector);
}

// Inverse transformation
AffineTransform AffineTransform::GetInverse() const {
  if (!HasInverse()) {
    throw std::runtime_error("Transform is not invertible");
  }

  AffineTransform inverse(m_dof);
  auto itk_inverse = m_transform->GetInverseTransform();

  // Safe cast to AffineTransform
  auto affine_inverse =
      dynamic_cast<ITKTransformType *>(itk_inverse.GetPointer());
  if (affine_inverse) {
    inverse.SetFromITKTransform(affine_inverse);
  } else {
    throw std::runtime_error(
        "Failed to cast inverse transform to AffineTransform");
  }
  return inverse;
}

bool AffineTransform::HasInverse() const {
  MatrixType matrix = m_transform->GetMatrix();
  // Calculate determinant manually for 3x3 matrix
  double det = matrix[0][0] *
                   (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
               matrix[0][1] *
                   (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
               matrix[0][2] *
                   (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  return std::abs(det) > 1e-10; // Numerical tolerance for singularity
}

// Transform composition
AffineTransform AffineTransform::Compose(const AffineTransform &other) const {
  AffineTransform result(*this);
  result.ComposeWith(other);
  return result;
}

AffineTransform &AffineTransform::ComposeWith(const AffineTransform &other) {
  // ITK handles the composition
  m_transform->Compose(other.m_transform);
  UpdateParametersFromTransform();
  return *this;
}

// ITK compatibility
void AffineTransform::SetFromITKTransform(ITKTransformType::Pointer transform) {
  m_transform->SetParameters(transform->GetParameters());
  m_transform->SetFixedParameters(transform->GetFixedParameters());
  UpdateParametersFromTransform();
}

// File I/O - Basic format
bool AffineTransform::SaveToFile(const std::string &filename) const {
  try {
    auto writer = itk::TransformFileWriterTemplate<double>::New();
    writer->SetFileName(filename);
    writer->SetInput(m_transform);
    writer->Update();
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error saving transform: " << e.what() << std::endl;
    return false;
  }
}

bool AffineTransform::LoadFromFile(const std::string &filename) {
  try {
    auto reader = itk::TransformFileReaderTemplate<double>::New();
    reader->SetFileName(filename);
    reader->Update();

    auto transform_list = reader->GetTransformList();
    if (transform_list->empty()) {
      return false;
    }

    auto loaded_transform =
        dynamic_cast<ITKTransformType *>(transform_list->front().GetPointer());
    if (!loaded_transform) {
      return false;
    }

    SetFromITKTransform(loaded_transform);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error loading transform: " << e.what() << std::endl;
    return false;
  }
}

// FSL format compatibility
bool AffineTransform::SaveToFSLFormat(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  // FSL uses 4x4 matrix format
  MatrixType matrix = m_transform->GetMatrix();
  VectorType translation = m_transform->GetTranslation();

  file << std::fixed << std::setprecision(10);

  // Write 4x4 matrix in FSL format
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      file << matrix[i][j] << "  ";
    }
    file << translation[i] << std::endl;
  }
  file << "0  0  0  1" << std::endl;

  return true;
}

bool AffineTransform::LoadFromFSLFormat(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  MatrixType matrix;
  VectorType translation;

  // Read 4x4 matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!(file >> matrix[i][j])) {
        return false;
      }
    }
    if (!(file >> translation[i])) {
      return false;
    }
  }

  // Skip the last row (0 0 0 1)
  double dummy;
  for (int i = 0; i < 4; ++i) {
    file >> dummy;
  }

  // Set the transform
  m_transform->SetMatrix(matrix);
  m_transform->SetTranslation(translation);
  UpdateParametersFromTransform();

  return true;
}

// Transform quality assessment
AffineTransform::TransformQuality AffineTransform::AssessQuality() const {
  TransformQuality quality;

  MatrixType matrix = m_transform->GetMatrix();

  // Calculate determinant manually for 3x3 matrix
  quality.determinant =
      matrix[0][0] *
          (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
      matrix[0][1] *
          (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
      matrix[0][2] *
          (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  quality.preserves_orientation = quality.determinant > 0;
  quality.is_invertible = std::abs(quality.determinant) > 1e-10;

  // Calculate condition number (rough estimate using matrix norms)
  // For a 3x3 matrix, this is a simplified approximation
  double max_norm = 0.0, min_norm = std::numeric_limits<double>::max();
  for (int i = 0; i < 3; ++i) {
    double row_norm = 0.0;
    for (int j = 0; j < 3; ++j) {
      row_norm += matrix[i][j] * matrix[i][j];
    }
    row_norm = std::sqrt(row_norm);
    max_norm = std::max(max_norm, row_norm);
    min_norm = std::min(min_norm, row_norm);
  }
  quality.condition_number = (min_norm > 1e-10) ? max_norm / min_norm : 1e10;

  // Extract scaling factors
  quality.scaling_factors = ExtractScaling(matrix);

  // Calculate maximum shear angle (simplified)
  quality.max_shear_angle = 0.0; // TODO: Implement proper shear calculation

  return quality;
}

// Debug and display
void AffineTransform::Print(std::ostream &os) const {
  os << "AffineTransform (DOF: " << static_cast<int>(m_dof) << ")" << std::endl;
  os << "Matrix:" << std::endl;

  MatrixType matrix = m_transform->GetMatrix();
  VectorType translation = m_transform->GetTranslation();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      os << std::setw(12) << std::fixed << std::setprecision(6) << matrix[i][j]
         << " ";
    }
    os << std::setw(12) << translation[i] << std::endl;
  }
  os << "   0.000000    0.000000    0.000000    1.000000" << std::endl;

  // Show decomposition
  double rx, ry, rz;
  GetEulerAngles(rx, ry, rz);
  VectorType scaling = GetScaling();

  os << "Decomposition:" << std::endl;
  os << "  Rotation (deg): [" << rx << ", " << ry << ", " << rz << "]"
     << std::endl;
  os << "  Translation: [" << translation[0] << ", " << translation[1] << ", "
     << translation[2] << "]" << std::endl;
  os << "  Scaling: [" << scaling[0] << ", " << scaling[1] << ", " << scaling[2]
     << "]" << std::endl;
}

std::string AffineTransform::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

// Parameter validation and clamping
AffineTransform::ParametersType
AffineTransform::ClampParameters(const ParametersType &params,
                                 DegreesOfFreedom dof) {
  ParametersType clamped = params;

  // Clamp rotation angles to [-180, 180] degrees
  if (static_cast<int>(dof) >= 6) {
    for (int i = 3; i < 6; ++i) {
      if (i < clamped.size()) {
        while (clamped[i] > 180.0)
          clamped[i] -= 360.0;
        while (clamped[i] < -180.0)
          clamped[i] += 360.0;
      }
    }
  }

  // Clamp scaling factors to reasonable range [0.1, 10.0]
  if (static_cast<int>(dof) >= 7) {
    int start_idx = (static_cast<int>(dof) == 7) ? 6 : 6;
    int end_idx = (static_cast<int>(dof) == 7) ? 7 : 9;

    for (int i = start_idx; i < end_idx && i < clamped.size(); ++i) {
      clamped[i] = std::max(0.1, std::min(10.0, clamped[i]));
    }
  }

  return clamped;
}

bool AffineTransform::ValidateParameters(const ParametersType &params,
                                         DegreesOfFreedom dof) {
  int expected_size = static_cast<int>(dof);

  if (params.size() != expected_size) {
    return false;
  }

  // Check for NaN or infinite values
  for (double param : params) {
    if (!std::isfinite(param)) {
      return false;
    }
  }

  return true;
}

// Internal parameter mapping functions
void AffineTransform::UpdateTransformFromParameters() {
  MapParametersToTransform(m_parameters);
}

void AffineTransform::UpdateParametersFromTransform() {
  m_parameters = MapTransformToParameters();
}

void AffineTransform::MapParametersToTransform(const ParametersType &params) {
  // Parameters layout:
  // [tx, ty, tz, rx, ry, rz, sx, sy, sz, shxy, shxz, shyz]
  // Depending on DOF, we use different subsets

  VectorType translation;
  translation.Fill(0.0);

  double rx = 0.0, ry = 0.0, rz = 0.0;
  double sx = 1.0, sy = 1.0, sz = 1.0;

  // Translation (always first 3 parameters if available)
  if (params.size() >= 3) {
    translation[0] = params[0];
    translation[1] = params[1];
    translation[2] = params[2];
  }

  // Rotation (parameters 3-5)
  if (params.size() >= 6) {
    rx = DegToRad(params[3]);
    ry = DegToRad(params[4]);
    rz = DegToRad(params[5]);
  }

  // Scaling
  if (m_dof == DegreesOfFreedom::Similarity && params.size() >= 7) {
    // Uniform scaling
    sx = sy = sz = params[6];
  } else if (m_dof == DegreesOfFreedom::Affine && params.size() >= 9) {
    // Non-uniform scaling
    sx = params[6];
    sy = params[7];
    sz = params[8];
  }

  // Create transformation matrix
  MatrixType rotation = CreateRotationMatrix(rx, ry, rz);
  MatrixType scaling = CreateScalingMatrix(sx, sy, sz);
  MatrixType matrix = rotation * scaling;

  // TODO: Add shear parameters for full 12-DOF affine
  // For now, we implement up to 9-DOF (translation + rotation + scaling)

  m_transform->SetMatrix(matrix);
  m_transform->SetTranslation(translation);
}

AffineTransform::ParametersType
AffineTransform::MapTransformToParameters() const {
  ParametersType params(GetNumberOfParameters(), 0.0);

  VectorType translation = m_transform->GetTranslation();
  MatrixType matrix = m_transform->GetMatrix();

  // Translation
  if (params.size() >= 3) {
    params[0] = translation[0];
    params[1] = translation[1];
    params[2] = translation[2];
  }

  // Extract scaling and rotation
  VectorType scaling = ExtractScaling(matrix);

  // Normalize matrix to get pure rotation
  MatrixType rotation_matrix = matrix;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rotation_matrix[i][j] /= scaling[i];
    }
  }

  // Decompose rotation
  double rx, ry, rz;
  DecomposeRotationMatrix(rotation_matrix, rx, ry, rz);

  // Rotation parameters
  if (params.size() >= 6) {
    params[3] = RadToDeg(rx);
    params[4] = RadToDeg(ry);
    params[5] = RadToDeg(rz);
  }

  // Scaling parameters
  if (m_dof == DegreesOfFreedom::Similarity && params.size() >= 7) {
    params[6] = (scaling[0] + scaling[1] + scaling[2]) / 3.0; // Average scaling
  } else if (m_dof == DegreesOfFreedom::Affine && params.size() >= 9) {
    params[6] = scaling[0];
    params[7] = scaling[1];
    params[8] = scaling[2];
  }

  return params;
}

// Mathematical helper functions
AffineTransform::MatrixType
AffineTransform::CreateRotationMatrix(double rx, double ry, double rz) {
  // ZYX Euler angle convention (common in medical imaging)
  double cx = std::cos(rx), sx = std::sin(rx);
  double cy = std::cos(ry), sy = std::sin(ry);
  double cz = std::cos(rz), sz = std::sin(rz);

  MatrixType matrix;

  // Combined rotation matrix (Rz * Ry * Rx)
  matrix[0][0] = cy * cz;
  matrix[0][1] = -cy * sz;
  matrix[0][2] = sy;

  matrix[1][0] = sx * sy * cz + cx * sz;
  matrix[1][1] = -sx * sy * sz + cx * cz;
  matrix[1][2] = -sx * cy;

  matrix[2][0] = -cx * sy * cz + sx * sz;
  matrix[2][1] = cx * sy * sz + sx * cz;
  matrix[2][2] = cx * cy;

  return matrix;
}

void AffineTransform::DecomposeRotationMatrix(const MatrixType &matrix,
                                              double &rx, double &ry,
                                              double &rz) {
  // Extract Euler angles from rotation matrix (ZYX convention)
  ry = std::asin(std::max(-1.0, std::min(1.0, matrix[0][2])));

  if (std::abs(matrix[0][2]) < 0.99999) {
    rx = std::atan2(-matrix[1][2], matrix[2][2]);
    rz = std::atan2(-matrix[0][1], matrix[0][0]);
  } else {
    // Gimbal lock case
    rx = std::atan2(matrix[2][1], matrix[1][1]);
    rz = 0.0;
  }
}

AffineTransform::MatrixType
AffineTransform::CreateScalingMatrix(double sx, double sy, double sz) {
  MatrixType matrix;
  matrix.SetIdentity();
  matrix[0][0] = sx;
  matrix[1][1] = sy;
  matrix[2][2] = sz;
  return matrix;
}

AffineTransform::VectorType
AffineTransform::ExtractScaling(const MatrixType &matrix) {
  VectorType scaling;

  // Calculate the length of each column vector
  for (int i = 0; i < 3; ++i) {
    double sum = 0.0;
    for (int j = 0; j < 3; ++j) {
      sum += matrix[j][i] * matrix[j][i];
    }
    scaling[i] = std::sqrt(sum);
  }

  return scaling;
}

// Operators
std::ostream &operator<<(std::ostream &os, const AffineTransform &transform) {
  transform.Print(os);
  return os;
}

bool operator==(const AffineTransform &lhs, const AffineTransform &rhs) {
  if (lhs.GetDegreesOfFreedom() != rhs.GetDegreesOfFreedom()) {
    return false;
  }

  auto params1 = lhs.GetParameters();
  auto params2 = rhs.GetParameters();

  if (params1.size() != params2.size()) {
    return false;
  }

  const double tolerance = 1e-6;
  for (size_t i = 0; i < params1.size(); ++i) {
    if (std::abs(params1[i] - params2[i]) > tolerance) {
      return false;
    }
  }

  return true;
}

bool operator!=(const AffineTransform &lhs, const AffineTransform &rhs) {
  return !(lhs == rhs);
}