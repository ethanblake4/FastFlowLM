/// \file system_prompt_cache.hpp
/// \brief Validation and token boundaries for independent prefix-cached requests.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <nlohmann/json.hpp>

namespace flm {
// Validate before message normalization: silently merging extra turns would hide
// a client's accidental conversation history. Media belongs in the user turn.
inline void validate_system_prompt_cache_request(const nlohmann::ordered_json& messages) {
    if (!messages.is_array() || messages.size() != 2 ||
        messages[0].value("role", "") != "system" ||
        messages[1].value("role", "") != "user") {
        throw std::invalid_argument("cache_system_prompt requires exactly one system message followed by one user message.");
    }
    for (const auto* field : {"images", "audios"}) {
        if (messages[0].contains(field) && !messages[0][field].empty()) {
            throw std::invalid_argument("cache_system_prompt requires a text-only system message.");
        }
    }
    const auto& content = messages[0].at("content");
    if (content.is_string()) return;
    if (content.is_array() && std::all_of(content.begin(), content.end(), [](const auto& part) {
        return part.is_object() && part.value("type", "") == "text" &&
            part.contains("text") && part["text"].is_string();
    })) return;
    throw std::invalid_argument("cache_system_prompt requires a text-only system message.");
}

// A template may fold the system message into the user turn, and tokenization
// can merge across their boundary. Only pin tokens common to both probes AND
// the actual request. Always leave a real suffix to prefill after restore.
inline std::vector<int> system_prompt_prefix(const std::vector<int>& a,
                                             const std::vector<int>& b,
                                             const std::vector<int>& request) {
    const auto limit = std::min({a.size(), b.size(), request.empty() ? size_t{0} : request.size() - 1});
    size_t count = 0;
    while (count < limit && a[count] == b[count] && a[count] == request[count]) ++count;
    return {a.begin(), a.begin() + count};
}
} // namespace flm
