import matplotlib.pyplot as plt

threads = [1, 2, 4, 6, 8]
times = [3.375, 1.947, 1.093, 0.863, 0.652]

# Calculate speedup
speedup = [times[0] / t for t in times]

plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, 'o-', label='Measured Speedup')
plt.plot(threads, threads, '--', color='gray', label='Ideal Linear Speedup (8x)')

plt.title('Ising Model OpenMP Scaling (AMD Ryzen 7 5850U)')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup (vs 1 Thread)')
plt.xticks(threads)
plt.legend()
plt.grid(True)
plt.savefig('speedup_plot.png', dpi=300)
print("Plot saved as speedup_plot.png")
