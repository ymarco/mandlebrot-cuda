#include "BmpSave.cuh"
#include <fstream>
#include <string>
namespace BmpSave {
__host__ __device__ Color::Color(unsigned char rr, unsigned char gg,
                                 unsigned char bb)
    : r(rr), g(gg), b(bb){};

__host__ __device__ Color::Color(int col) {
  r = (col & 0xFF0000) >> 16;
  g = (col & 0x00FF00) >> 8;
  b = col & 0x0000FF;
}
__host__ __device__ Color::Color() {
  r = 0;
  g = 0;
  b = 0;
}

void saveBmp(const std::string &filename, const int width, const int height,
             Color *pixels) {
  BmpHeader header(width, height);
  std::ofstream fout(filename.c_str(), std::ios::binary);
  fout.write(reinterpret_cast<char *>(&header), sizeof(BmpHeader));
  fout.write(reinterpret_cast<char *>(pixels), width * height * sizeof(Color));
}

} // namespace BmpSave
