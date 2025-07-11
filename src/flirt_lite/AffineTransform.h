#ifndef AFFINE_TRANSFORM_H
#define AFFINE_TRANSFORM_H

#include <iostream>
#include <string>
#include <vector>

// ITK Headers
#include "itkAffineTransform.h"
#include "itkMatrix.h"
#include "itkPoint.h"
#include "itkVector.h"

// Forward declarations
namespace itk {
template <typename TScalar, unsigned int NDimensions> class AffineTransform;
}

class AffineTransform {
public:
  using ScalarType = double;
  using ITKTransformType = itk::AffineTransform<ScalarType, 3>;
  using MatrixType = itk::Matrix<ScalarType, 3, 3>;
  using VectorType = itk::Vector<ScalarType, 3>;
  using PointType = itk::Point<ScalarType, 3>;
  using ParametersType = std::vector<ScalarType>;

  // Transform degrees of freedom enumeration
  enum class DegreesOfFreedom {
    RigidBody = 6,  // 3 translations + 3 rotations
    Similarity = 7, // Rigid body + uniform scaling
    Affine = 12     // Full affine transformation
  };

private:
  ITKTransformType::Pointer m_transform;
  DegreesOfFreedom m_dof;
  ParametersType m_parameters;

public:
  // Constructor
  AffineTransform(DegreesOfFreedom dof = DegreesOfFreedom::Affine);
  AffineTransform(const AffineTransform &other);
  ~AffineTransform() = default;

  // Assignment operation
  AffineTransform &operator=(const AffineTransform &other);

  // Parameter setting and retrieval
  void SetParameters(const ParametersType &params);
  ParametersType GetParameters() const;
  int GetNumberOfParameters() const;

  // Transform degrees of freedom
  void SetDegreesOfFreedom(DegreesOfFreedom dof);
  DegreesOfFreedom GetDegreesOfFreedom() const { return m_dof; }

  // Initialization methods
  void SetIdentity();
  void SetCenter(const PointType &center);
  PointType GetCenter() const;

  // Set transform from Euler angles and translation
  void SetFromEulerAngles(double rx, double ry, double rz, double tx, double ty,
                          double tz, double sx = 1.0, double sy = 1.0,
                          double sz = 1.0);

  // Get Euler angle decomposition
  void GetEulerAngles(double &rx, double &ry, double &rz) const;
  VectorType GetTranslation() const;
  VectorType GetScaling() const;

  // Matrix operations
  void SetMatrix(const MatrixType &matrix);
  MatrixType GetMatrix() const;
  void SetTranslation(const VectorType &translation);

  // Point transformation
  PointType TransformPoint(const PointType &point) const;
  VectorType TransformVector(const VectorType &vector) const;

  // Inverse transformation
  AffineTransform GetInverse() const;
  bool HasInverse() const;

  // Transform composition
  AffineTransform Compose(const AffineTransform &other) const;
  AffineTransform &ComposeWith(const AffineTransform &other);

  // ITK compatibility
  ITKTransformType::Pointer GetITKTransform() const { return m_transform; }
  void SetFromITKTransform(ITKTransformType::Pointer transform);

  // File I/O
  bool SaveToFile(const std::string &filename) const;
  bool LoadFromFile(const std::string &filename);

  // FSL format compatibility
  bool SaveToFSLFormat(const std::string &filename) const;
  bool LoadFromFSLFormat(const std::string &filename);

  // Transform quality assessment
  struct TransformQuality {
    double determinant; // Determinant, detect flipping
    double
        condition_number; // Condition number, detect ill-conditioned transforms
    bool is_invertible;   // Whether invertible
    bool preserves_orientation; // Whether orientation is preserved
    VectorType scaling_factors; // Scaling factors
    double max_shear_angle;     // Maximum shear angle (degrees)
  };

  TransformQuality AssessQuality() const;

  // Debug and display
  void Print(std::ostream &os = std::cout) const;
  std::string ToString() const;

  // Parameter boundary checking
  static ParametersType ClampParameters(const ParametersType &params,
                                        DegreesOfFreedom dof);
  static bool ValidateParameters(const ParametersType &params,
                                 DegreesOfFreedom dof);

private:
  // Internal helper functions
  void UpdateTransformFromParameters();
  void UpdateParametersFromTransform();

  // Parameter mapping (depends on DOF)
  void MapParametersToTransform(const ParametersType &params);
  ParametersType MapTransformToParameters() const;

  // Mathematical helper functions
  static MatrixType CreateRotationMatrix(double rx, double ry, double rz);
  static void DecomposeRotationMatrix(const MatrixType &matrix, double &rx,
                                      double &ry, double &rz);
  static MatrixType CreateScalingMatrix(double sx, double sy, double sz);
  static VectorType ExtractScaling(const MatrixType &matrix);

  // Angle unit conversion
  static double DegToRad(double degrees) { return degrees * M_PI / 180.0; }
  static double RadToDeg(double radians) { return radians * 180.0 / M_PI; }
};

// Stream operator overload
std::ostream &operator<<(std::ostream &os, const AffineTransform &transform);

// Comparison operators
bool operator==(const AffineTransform &lhs, const AffineTransform &rhs);
bool operator!=(const AffineTransform &lhs, const AffineTransform &rhs);

#endif // AFFINE_TRANSFORM_H