#include <include/get_nel.hpp>
int get_nel(const std::vector<double>& D, const std::vector<double>& S) {
    int n = D.size();
    double nel = lawrap::dot(n, D.data(), 1, S.data(), 1);
    return (int)std::round(nel);
}
