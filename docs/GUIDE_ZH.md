# NeuroCompass 用户指南

**精准导航神经运动** 🧭

本指南提供NeuroCompass运动校正工具的完整使用说明。

## 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [运动校正](#运动校正)
- [使用示例](#使用示例)
- [故障排除](#故障排除)
- [高级用法](#高级用法)

---

## 快速开始

1. **编译项目**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

2. **运行运动校正**:
   ```bash
   ./neurocompass_motion 你的4D数据.nii.gz
   ```

3. **查看结果**:
   - `motion_parameters.par`: 每个体积的运动参数
   - 控制台输出: 质量评估和统计信息

---

## 安装

### 系统要求
- C++17兼容编译器 (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16或更高版本
- 标准系统库

### 编译步骤
```bash
git clone https://github.com/Ruri-jiale/NeuroCompass.git
cd NeuroCompass
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 安装
```bash
sudo make install
```

---

## 运动校正

### 基本用法
```bash
neurocompass_motion 输入4D数据.nii.gz
```

### 输出说明

**控制台输出示例**:
```
NeuroCompass Motion Correction
==============================
Lightweight 4D medical image processing

图像维度: 144x144x60x57
体素大小: 1.5x1.5x2.0 mm
处理时间: 2.31 秒

运动统计:
平均框架位移: 0.101 mm
最大框架位移: 0.199 mm
质量等级: 优秀
```

**输出文件** (`motion_parameters.par`):
```
# 格式: tx ty tz rx ry rz 相似性
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
0.022718 0.076145 0.106310 0.000227 0.000761 0.001063 0.548937
...
```

### 质量等级
- **优秀**: 平均FD < 0.2 mm ⭐⭐⭐⭐⭐
- **良好**: 平均FD < 0.5 mm ⭐⭐⭐⭐
- **一般**: 平均FD < 1.0 mm ⭐⭐⭐
- **较差**: 平均FD > 1.0 mm ⭐

---

## 使用示例

### 示例1: 基本处理
```bash
# 处理单个4D文件
neurocompass_motion fmri数据.nii.gz

# 检查质量
grep "质量等级" motion_parameters.par
```

### 示例2: 批量处理
```bash
# 处理多个被试
for subject in sub-*/func/*.nii.gz; do
    echo "处理中: $subject"
    neurocompass_motion "$subject"
    # 移动结果到被试目录
    subject_dir=$(dirname "$subject")
    mv motion_parameters.par "${subject_dir}/motion_params.par"
done
```

### 示例3: 质量控制
```bash
# 提取所有被试的平均FD
for subject in sub-*; do
    if [ -f "${subject}/func/motion_params.par" ]; then
        mean_fd=$(grep "平均FD" "${subject}/func/motion_params.par" | awk '{print $4}')
        echo "${subject}: ${mean_fd} mm"
    fi
done
```

---

## 故障排除

### 常见问题

1. **"无法打开文件"错误**:
   - 检查文件路径和权限
   - 确保文件是有效的NIfTI格式 (.nii 或 .nii.gz)
   - 验证文件未损坏

2. **"体积数量不足"错误**:
   - 输入必须是至少包含2个体积的4D数据
   - 检查图像维度

3. **运动校正质量差**:
   - 高运动数据可能需要人工检查
   - 考虑从分析中排除高运动体积
   - 检查采集参数

4. **编译错误**:
   - 确保C++17支持: `gcc --version` (需要7+)
   - 更新CMake: `cmake --version` (需要3.16+)
   - 安装缺失的依赖项

### 性能提示
- 使用SSD存储获得更好的I/O性能
- 在本地驱动器而非网络存储上处理文件
- 考虑对多个被试进行并行处理

---

## 高级用法

### 自定义参数 (库使用)
```cpp
#include "StandaloneMCFLIRT.h"
using namespace neurocompass::standalone;

// 读取图像
auto image_data = StandaloneMCFLIRT::ReadNIfTI("输入.nii.gz");

// 执行运动校正
auto result = StandaloneMCFLIRT::CorrectMotion(image_data);

// 访问详细结果
for (const auto& motion : result.motion_params) {
    std::cout << "体积 " << motion.volume_index 
              << ": 平移=" << motion.params[0] 
              << "," << motion.params[1] 
              << "," << motion.params[2] << " mm" << std::endl;
}
```

### 与其他工具集成
```bash
# 示例: 与其他工具的集成
fslinfo 输入.nii.gz
neurocompass_motion 输入.nii.gz
# 使用其他工具应用变换...
```

---

## 支持

- **文档**: 查看 [docs/](../docs/) 目录获取详细指南
- **问题**: 在 [GitHub Issues](https://github.com/Ruri-jiale/NeuroCompass/issues) 上报告错误和功能请求
- **社区**: 加入讨论获取帮助和更新

---

*最后更新: 2025 | NeuroCompass v1.0*