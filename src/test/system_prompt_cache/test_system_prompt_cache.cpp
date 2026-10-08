#include "system_prompt_cache.hpp"
#include <iostream>

using json = nlohmann::ordered_json;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static bool accepts(const json& messages) {
    try { flm::validate_system_prompt_cache_request(messages); return true; }
    catch (const std::exception&) { return false; }
}

int main() {
    try {
        const json system = {{"role", "system"}, {"content", "Stable instructions"}};
        const json user = {{"role", "user"}, {"content", "New observation"}};
        require(accepts(json::array({system, user})), "single-turn text rejected");
        require(accepts(json::array({system, {{"role", "user"}, {"content", json::array({
            {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,..."}}}}
        })}}})), "user media rejected");
        require(accepts(json::array({{{"role", "system"}, {"content", json::array({
            {{"type", "text"}, {"text", "Instructions"}}, {{"type", "text"}, {"text", "Reference data"}}
        })}}, user})), "system text parts rejected");
        require(!accepts(json::array({user})), "missing system accepted");
        require(!accepts(json::array({user, system})), "wrong order accepted");
        require(!accepts(json::array({system, user, user})), "extra user turn accepted");
        require(!accepts(json::array({system, {{"role", "assistant"}, {"content", "Old result"}}, user})), "history accepted");
        require(!accepts(json::array({{{"role", "system"}, {"content", json::array({
            {{"type", "image_url"}, {"image_url", {{"url", "data:..."}}}}
        })}}, user})), "system media accepted");
        require(!accepts(json::array({{{"role", "system"}}, user})), "missing content accepted");
        require(!accepts(json::array({{{"role", "system"}, {"content", "text"}, {"images", json::array({"pixels"})}}, user})), "legacy system media accepted");
        require(!accepts(json::object()), "non-array accepted");

        require(flm::system_prompt_prefix({1,2,3,4}, {1,2,3,5}, {1,2,3,6,7}) == std::vector<int>({1,2,3}), "probe boundary incorrect");
        require(flm::system_prompt_prefix({1,2,3}, {1,2,4}, {1,9,5}) == std::vector<int>({1}), "actual tokenization not checked");
        require(flm::system_prompt_prefix({1,2}, {1,2}, {1,2}) == std::vector<int>({1}), "no suffix left to prefill");
        require(flm::system_prompt_prefix({1}, {2}, {1,3}).empty(), "unrelated probes cached");
        require(flm::system_prompt_prefix({1}, {1}, {}).empty(), "empty request cached");
        std::cout << "System prompt cache validation and token boundaries passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
