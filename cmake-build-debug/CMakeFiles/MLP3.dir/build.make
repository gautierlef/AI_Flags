# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\gautier\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.6693.114\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\gautier\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.6693.114\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\gautier\CLionProjects\MLP_Rattrapage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/MLP3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MLP3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MLP3.dir/flags.make

CMakeFiles/MLP3.dir/library.cpp.obj: CMakeFiles/MLP3.dir/flags.make
CMakeFiles/MLP3.dir/library.cpp.obj: ../library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MLP3.dir/library.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\MLP3.dir\library.cpp.obj -c C:\Users\gautier\CLionProjects\MLP_Rattrapage\library.cpp

CMakeFiles/MLP3.dir/library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MLP3.dir/library.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\gautier\CLionProjects\MLP_Rattrapage\library.cpp > CMakeFiles\MLP3.dir\library.cpp.i

CMakeFiles/MLP3.dir/library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MLP3.dir/library.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\gautier\CLionProjects\MLP_Rattrapage\library.cpp -o CMakeFiles\MLP3.dir\library.cpp.s

# Object files for target MLP3
MLP3_OBJECTS = \
"CMakeFiles/MLP3.dir/library.cpp.obj"

# External object files for target MLP3
MLP3_EXTERNAL_OBJECTS =

libMLP3.dll: CMakeFiles/MLP3.dir/library.cpp.obj
libMLP3.dll: CMakeFiles/MLP3.dir/build.make
libMLP3.dll: CMakeFiles/MLP3.dir/linklibs.rsp
libMLP3.dll: CMakeFiles/MLP3.dir/objects1.rsp
libMLP3.dll: CMakeFiles/MLP3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libMLP3.dll"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\MLP3.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MLP3.dir/build: libMLP3.dll

.PHONY : CMakeFiles/MLP3.dir/build

CMakeFiles/MLP3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\MLP3.dir\cmake_clean.cmake
.PHONY : CMakeFiles/MLP3.dir/clean

CMakeFiles/MLP3.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\gautier\CLionProjects\MLP_Rattrapage C:\Users\gautier\CLionProjects\MLP_Rattrapage C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug C:\Users\gautier\CLionProjects\MLP_Rattrapage\cmake-build-debug\CMakeFiles\MLP3.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MLP3.dir/depend

