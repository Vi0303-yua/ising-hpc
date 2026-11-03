#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h> // OpenMP header

// Notice we pass the RNG state as a pointer so each thread has its own!
static inline double rnd(uint64_t *state){
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (double)(*state >> 11) * (1.0/9007199254740992.0);
}
static inline int wrap(int a, int L){ return ((a % L) + L) % L; }

void run_ising(int L, double T, long neq, long nmeas, uint64_t seed){
    uint64_t rng_state = seed;
    const int N = L*L;
    std::vector<signed char> s(N);
    for(int i=0;i<N;i++) s[i] = (rnd(&rng_state) < 0.5) ? -1 : 1;

    const double beta = 1.0/T;
    double w[5];
    for(int j=0;j<5;j++){ int k = 2*(j-2); w[j] = std::exp(-beta*2.0*k); }

    double sE=0, sE2=0, sM=0, sM2=0;
    long cnt=0;

    for(long sweep=0; sweep<neq+nmeas; sweep++){
        for(int n=0; n<N; n++){
            int i = (int)(rnd(&rng_state)*N);
            int x = i % L, y = i / L;
            int nb = s[wrap(x+1,L)+y*L] + s[wrap(x-1,L)+y*L]
                   + s[x+wrap(y+1,L)*L] + s[x+wrap(y-1,L)*L];
            int k = (int)s[i] * nb;
            if(k <= 0 || rnd(&rng_state) < w[k/2+2]) s[i] = (signed char)(-s[i]);
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
    
    // Print the result for this temperature
    int thread_id = omp_get_thread_num();
    std::printf("Thread %d | T=%.3f | m=%.4f\n", thread_id, T, sM/cnt);
}

int main(){
    int L = 64;
    long neq = 1000;
    long nmeas = 5000; // Small so it finishes quickly
    int num_temps = 8;
    double temps[] = {1.8, 2.0, 2.1, 2.2, 2.269, 2.4, 2.6, 3.0};

    auto t0 = std::chrono::steady_clock::now();

    // THIS IS THE OPENMP MAGIC:
    #pragma omp parallel for
    for(int i=0; i<num_temps; i++){
        run_ising(L, temps[i], neq, nmeas, 12345 + i);
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1-t0).count();
    std::printf("Total wall time: %.3f seconds\n", secs);
    return 0;
}
