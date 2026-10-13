#include "async/common.h"
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Config cfg;
    cfg.N = std::atoi(argv[1]);
    if (!parse_source(argv[2], cfg.source)) return 2;
    cfg.r = std::atof(argv[3]);
    Problem p = init_problem(cfg);
    std::fprintf(stderr, "N=%d r=%.9g denom=%.9g\n", p.N, p.r, p.denom);
    std::fwrite(p.u0.data(), sizeof(float), p.u0.size(), stdout);
    std::fwrite(p.src.data(), sizeof(float), p.src.size(), stdout);
}
