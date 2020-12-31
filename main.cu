#include "BmpSave.cuh"
#include "mandelbrot.cuh"
#include <iostream>
#include <string>

__global__ void paint_mandelbrot(BmpSave::Color *img) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y;
  if (i >= I->RES_WIDTH)
    return;
  // img[i+j] = BmpSave::Color(50,50,50);
  BmpSave::Color clr = MColor(i, j);
  img[I->RES_WIDTH * j + i] = clr;
}
cudaError_t s;
#define cuError(cmd)                                                           \
  s = cmd;                                                                     \
  if (s != cudaSuccess) {                                                      \
    std::cout << __LINE__ << " - " << cudaGetErrorString(s) << std::endl;      \
  }

int main(int argc, char *argv[]) {
  // initializing cuda mem
  cudaFree(0);
  // args parsing:
  std::cout << "given arguments: " << argc << "\n";
  if (argc != 4 && argc != 6) {
    std::cerr << "incorrect amount of arguments.\nusage: " << argv[0]
              << " center_x center_y zoom [img_width, image_height]\n";
    return -1;
  }

  ImgInfo img_info;
  img_info.CENTER_X = std::stod(argv[1]);
  img_info.CENTER_Y = std::stod(argv[2]);
  img_info.ZOOM = std::stoi(argv[3]);
  if (argc == 6) {
    img_info.RES_WIDTH = std::stoi(argv[4]);
    img_info.RES_HEIGHT = std::stoi(argv[5]);
  } else {
    img_info.RES_WIDTH = 2560;
    img_info.RES_HEIGHT = 1440;
  }
  // initializing the __constant__ I
  ImgInfo *I_addr;
  // cuError(cudaGetSymbolAddress((void**)&I_addr, I));
  cuError(cudaMemcpyToSymbol(I, &img_info, sizeof(ImgInfo)));
  // std::cout << rerange(0,0) << "," <<rerange(RES_WIDTH, RES_HEIGHT) <<
  // ",center = " << rerange(RES_WIDTH/2, RES_HEIGHT/2) << std::endl;
  int IMG_SIZE =
      img_info.RES_WIDTH * img_info.RES_HEIGHT * sizeof(BmpSave::Color);

  BmpSave::Color *d_img = nullptr;
  cuError(cudaMalloc(&d_img, IMG_SIZE));

  // moving img_info to device:
  dim3 block(256);
  dim3 grid((img_info.RES_WIDTH + block.x - 1) / block.x, img_info.RES_HEIGHT);

  std::cout << d_img << "- d_img" << std::endl;

  paint_mandelbrot<<<grid, block>>>(d_img);

  cuError(cudaGetLastError());
  BmpSave::Color *h_img =
      new BmpSave::Color[img_info.RES_HEIGHT * img_info.RES_WIDTH];
  std::cout << h_img << std::endl;
  cuError(cudaMemcpy(h_img, d_img, IMG_SIZE, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  cudaFree(d_img);
  std::string filename = "images/" + std::to_string(img_info.CENTER_X) + "," +
                         std::to_string(img_info.CENTER_Y) +
                         ",zoom=" + std::to_string(img_info.ZOOM) + ".bmp";
  BmpSave::saveBmp(filename, img_info.RES_WIDTH, img_info.RES_HEIGHT, h_img);
  delete[] h_img;
}
