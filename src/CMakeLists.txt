# Core modules
add_subdirectory(io)
# Temporarily disable modules with compilation issues
# add_subdirectory(flirt_lite)
# add_subdirectory(bet_lite)  
# add_subdirectory(mcflirt_lite)

# Standalone implementation (no external dependencies) - WORKING
add_subdirectory(standalone)

# Validation module (currently disabled due to dependency issues)
# add_subdirectory(validation) 