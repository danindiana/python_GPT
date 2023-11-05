#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

unsigned int ScanDirectory(const fs::path& directory) {
    unsigned int file_count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (fs::is_regular_file(entry)) {
            std::cout << entry.path().string() << std::endl;
            ++file_count;
        }
    }
    return file_count;
}

int main() {
    std::string target_directory;
    std::cout << "Please enter the target directory to scan: ";
    std::getline(std::cin, target_directory);

    if (!fs::exists(target_directory) || !fs::is_directory(target_directory)) {
        std::cerr << "Error: The specified path does not exist or is not a directory." << std::endl;
        return 1;
    }

    unsigned int total_files = ScanDirectory(target_directory);

    std::cout << "The scan is complete. Total number of files scanned: " << total_files << std::endl;

    return 0;
}
