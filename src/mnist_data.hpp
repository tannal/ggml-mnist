#pragma once

#include <vector>
#include <string>

class MNISTData {
public:
    MNISTData(const std::string& images_file, const std::string& labels_file);

    size_t size() const { return labels.size(); }
    const std::vector<float>& get_image(size_t index) const { return images[index]; }
    int get_label(size_t index) const { return labels[index]; }

private:
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    void load_images(const std::string& filename);
    void load_labels(const std::string& filename);
};