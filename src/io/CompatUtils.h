#ifndef COMPAT_UTILS_H
#define COMPAT_UTILS_H

#include <string>

namespace neurocompass {
namespace io {
namespace compat {

// C++17 compatible string ends_with function
inline bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

// C++17 compatible string starts_with function (for future use)
inline bool starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.length() > str.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

} // namespace compat
} // namespace io
} // namespace neurocompass

#endif // COMPAT_UTILS_H