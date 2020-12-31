#ifndef BMPSAVE_H
#define BMPSAVE_H

#include <fstream>
#include <vector>

namespace BmpSave {

struct Color {
  unsigned char b;
  unsigned char g;
  unsigned char r;
  unsigned char a;
  __host__ __device__ Color(unsigned char rr, unsigned char gg,
                            unsigned char bb);
  __host__ __device__ Color(int col);
  __host__ __device__ Color();
};

#pragma pack(push, 2)
struct BmpHeader {
  int16_t magic = 0x4D42;
  int fileSize;
  int16_t reserved1 = 0;
  int16_t reserved2 = 0;
  int pixelArrayOffset = sizeof(BmpHeader);
  int infoHeaderSize = 40;
  int width;
  int height;
  int16_t planes = 1;
  int16_t bitsPerColor = 32;
  int compression = 0;
  int imageSize = 0;
  int horizontalResolution = 0;
  int verticalResolution = 0;
  int colors = 0;
  int importantColors = 0;
  BmpHeader(int a, int b) {
    width = a;
    height = b;
    fileSize = sizeof(BmpHeader) + sizeof(Color) * width * height;
  }
};
#pragma pack(pop)

void saveBmp(const std::string &filename, const int width, const int height,
             Color *pixels);
} // namespace BmpSave

#endif
