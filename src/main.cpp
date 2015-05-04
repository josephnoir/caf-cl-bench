
#include "config.hpp"

#include "caf/all.hpp"
#include "caf/opencl/spawn_cl.hpp"

using namespace std;
using namespace caf;

namespace {

constexpr const char* kernel_source = R"__(
  __kernel void mandelbrot(__global float* config,
                           __global int* output) {
    unsigned iterations = config[0];
    unsigned width = config[1];
    unsigned height = config[2];
    float min_re = config[3];
    float max_re = config[4];
    float min_im = config[5];
    float max_im = config[6];
    float re_factor = (max_re-min_re)/(width-1);
    float im_factor = (max_im-min_im)/(height-1);
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    float z_re = min_re + x*re_factor;
    float z_im = max_im - y*im_factor;
    float const_re = z_re;
    float const_im = z_im;
    unsigned cnt = 0;
    float cond = 0;
    do {
      float tmp_re = z_re;
      float tmp_im = z_im;
      z_re = ( tmp_re*tmp_re - tmp_im*tmp_im ) + const_re;
      z_im = ( 2 * tmp_re * tmp_im ) + const_im;
      cond = (z_re - z_im) * (z_re - z_im);
      cnt ++;
    } while (cnt < iterations && cond <= 4.0f);
    output[x+y*width] = cnt;
  }
)__";

} // namespace <anonymous>

void supervisor() {
  auto clworker = spawn_cl<int*(float*)>(clprog, "mandel", {width, height});
}

int main(int argc, char** argv) {
  announce<vector<float>>("float_vector");
  spawn(supervisor);
  await_all_actors_done();
  shutdown();
  return 0;
}
