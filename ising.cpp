#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>

static uint64_t rng_state = 88172645463325252ULL;
static inline double rnd(){
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0/9007199254740992.0);
}
static inline int wrap(int a, int L){ return ((a % L) + L) % L; }

int main(int argc, char **argv){
    int    L     = (argc>1) ? std::atoi(argv[1]) : 32;
    double T     = (argc>2) ? std::atof(argv[2]) : 2.269;
    long   neq   = (argc>3) ? std::atol(argv[3]) : 2000;
    long   nmeas = (argc>4) ? std::atol(argv[4]) : 20000;
    uint64_t seed= (argc>5) ? std::strtoull(argv[5],nullptr,10) : 12345ULL;

    rng_state = seed ? seed : 1ULL;

    const int N = L*L;
    std::vector<signed char> s(N);
    for(int i=0;i<N;i++) s[i] = (rnd() < 0.5) ? -1 : 1;

    const double beta = 1.0/T;
    double w[5];
    for(int j=0;j<5;j++){ int k = 2*(j-2); w[j] = std::exp(-beta*2.0*k); }

    double sE=0, sE2=0, sM=0, sM2=0;
    long cnt=0;

    auto t0 = std::chrono::steady_clock::now();

    for(long sweep=0; sweep<neq+nmeas; sweep++){
        for(int n=0; n<N; n++){
            int i = (int)(rnd()*N);
            int x = i % L, y = i / L;
            int nb = s[wrap(x+1,L)+y*L] + s[wrap(x-1,L)+y*L]
                   + s[x+wrap(y+1,L)*L] + s[x+wrap(y-1,L)*L];
            int k = (int)s[i] * nb;
            if(k <= 0 || rnd() < w[k/2+2]) s[i] = (signed char)(-s[i]);
        }
        if(sweep >= neq){
            int sum = 0;
            for(int i=0;i<N;i++) sum += s[i];
            double e = 0.0;
            for(int y=0;y<L;y++) for(int x=0;x<L;x++){
                int i = x + y*L;
                int nb = s[wrap(x+1,L)+y*L] + s[wrap(x-1,L)+y*L]
                       + s[x+wrap(y+1,L)*L] + s[x+wrap(y-1,L)*L];
                e -= 0.5 * s[i] * nb;
            }
            e /= N;
            double m = std::fabs((double)sum) / N;
            sE += e; sE2 += e*e; sM += m; sM2 += m*m; cnt++;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1-t0).count();

    double E  = sE/cnt,  E2 = sE2/cnt;
    double M  = sM/cnt,  M2 = sM2/cnt;
    double C  = beta*beta*N*(E2 - E*E);
    double chi= beta*N*(M2 - M*M);

    std::printf("L=%d T=%.6f E=%.8f m=%.8f C=%.6f chi=%.6f time=%.3fs sites/s=%.3e\n",
                L, T, E, M, C, chi, secs, (double)N*(neq+nmeas)/secs);
    return 0;
}
