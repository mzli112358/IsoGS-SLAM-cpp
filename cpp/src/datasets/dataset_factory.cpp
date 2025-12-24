#include "datasets/dataset_factory.hpp"
#include "datasets/replica_dataset.hpp"
#include "datasets/tum_dataset.hpp"
#include "datasets/scannet_dataset.hpp"
#include "datasets/scannetpp_dataset.hpp"
#include "datasets/replica_v2_dataset.hpp"
#include <algorithm>
#include <cctype>

namespace isogs {

std::unique_ptr<DatasetBase> DatasetFactory::create(
    const std::string& dataset_type,
    const std::string& basedir,
    const std::string& sequence,
    int start,
    int end,
    int stride,
    int desired_height,
    int desired_width)
{
    // Convert to lowercase
    std::string type_lower = dataset_type;
    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (type_lower == "replica") {
        return std::make_unique<ReplicaDataset>(
            basedir, sequence, start, end, stride, desired_height, desired_width);
    } else if (type_lower == "tum") {
        return std::make_unique<TUMDataset>(
            basedir, sequence, start, end, stride, desired_height, desired_width);
    } else if (type_lower == "scannet") {
        return std::make_unique<ScanNetDataset>(
            basedir, sequence, start, end, stride, desired_height, desired_width);
    } else if (type_lower == "scannetpp") {
        return std::make_unique<ScanNetPPDataset>(
            basedir, sequence, start, end, stride, desired_height, desired_width);
    } else if (type_lower == "replicav2" || type_lower == "replica_v2") {
        return std::make_unique<ReplicaV2Dataset>(
            basedir, sequence, start, end, stride, desired_height, desired_width);
    } else {
        throw std::runtime_error("Unknown dataset type: " + dataset_type);
    }
}

} // namespace isogs

