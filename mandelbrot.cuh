#ifndef MANDELBROT_H
#define MANDELBROT_H
#include "BmpSave.cuh"
#include <fstream>
#include <string>

struct ImgInfo {
  double CENTER_X;
  double CENTER_Y;
  int ZOOM;
  int RES_WIDTH /*= 2560*/;
  int RES_HEIGHT /*= 1440*/;
};

__constant__ ImgInfo I[1];

__device__ inline BmpSave::Color gen_color(unsigned short x) {
  BmpSave::Color c;
  c.b = 5 * x;
  c.g = 3 * x;
  c.r = 1 * x;
  return c;
};

struct Comp {
  double r;
  double i;
};

__device__ inline Comp rerange(unsigned int i, unsigned int j) {
  Comp x;
  x.r = (double(i) - I->RES_WIDTH / 2 + I->CENTER_X * I->RES_WIDTH) /
        I->RES_WIDTH;
  x.i = (double(j) - I->RES_HEIGHT / 2 + I->CENTER_Y * I->RES_HEIGHT) /
        I->RES_WIDTH;
  x.r = x.r / I->ZOOM + I->CENTER_X * (I->ZOOM - 1) / I->ZOOM;
  x.i = x.i / I->ZOOM + I->CENTER_Y * (I->ZOOM - 1) / I->ZOOM;
  return x;
}
__device__ inline unsigned short converges(Comp c) {
  Comp z({0, 0});
  double zr_new = 0;
  for (unsigned short i = 1; i < 1024; i++) {
    if (z.r * z.r + z.i * z.i > 4)
      return i;
    zr_new = z.r * z.r - z.i * z.i + c.r;
    z.i = +2 * z.i * z.r + c.i;
    z.r = zr_new;
  }
  return 0;
};

__device__ inline BmpSave::Color MColor(unsigned int i, unsigned int j) {
  Comp c = rerange(i, j);
  return gen_color(converges(c));
  // return BmpSave::Color(30,30,30);;
};

#endif
