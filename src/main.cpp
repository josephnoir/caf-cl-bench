
#include <chrono>

#include <QFile>
#include <QColor>
#include <QImage>
#include <QBuffer>
#include <QByteArray>

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
    float re_factor = (max_re - min_re) / (width - 1);
    float im_factor = (max_im - min_im) / (height - 1);
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    float z_re = min_re + x * re_factor;
    float z_im = max_im - y * im_factor;
    float const_re = z_re;
    float const_im = z_im;
    unsigned cnt = 0;
    float cond = 0;
    do {
      float tmp_re = z_re;
      float tmp_im = z_im;
      z_re = ( tmp_re * tmp_re - tmp_im * tmp_im ) + const_re;
      z_im = ( 2 * tmp_re * tmp_im ) + const_im;
      cond = (z_re - z_im) * (z_re - z_im);
      ++cnt;
    } while (cnt < iterations && cond <= 4.0f);
    output[x+y*width] = cnt;
  }
)__";

//#define PRINT_IMAGE
//#define ENABLE_DEBUG
#ifdef ENABLE_DEBUG
#define DEBUG(x) cerr << x << endl;
#else
#define DEBUG(x)
#endif

} // namespace <anonymous>

// !!!! ugly global values for time !!!!
unsigned long time_gpu = 0;
unsigned long time_cpu = 0;

inline void calculate_palette(std::vector<QColor>& storage, uint32_t iterations) {
  // generating new colors
  storage.clear();
  storage.reserve(iterations + 1);
  for (uint32_t i = 0; i < iterations; ++i) {
    QColor tmp;
    tmp.setHsv(((180.0 / iterations) * i) + 180.0, 255, 200);
    storage.push_back(tmp);
  }
  storage.push_back(QColor(qRgb(0,0,0)));
}

void color_and_print(const std::vector<QColor>& palette,
                     const std::vector<int>& counts,
                     uint32_t width, uint32_t height,
                     const char* identifier) {
  QImage image{static_cast<int>(width),
               static_cast<int>(height),
               QImage::Format_RGB32};
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      image.setPixel(x,y,palette[counts[x + y * width]].rgb());
    }
  }
  // print image
  QByteArray ba;
  QBuffer buf{&ba};
  buf.open(QIODevice::WriteOnly);
  image.save(&buf, image_format);
  buf.close();
  auto img = QImage::fromData(ba, image_format);
  std::ostringstream fname;
  fname << "mandelbrot_" << identifier;
  fname << image_file_ending;
  QFile f{fname.str().c_str()};
  if (!f.open(QIODevice::WriteOnly)) {
    DEBUG("could not open file: " << fname.str());
  } else {
    img.save(&f, image_format);
    DEBUG(identifier << " image saved");
  }
}

// calculates mandelbrot that contains the iteration count on GPU
void mandel_cl(event_based_actor* self,
               caf::opencl::program kernel,
               std::uint32_t iterations,
               std::uint32_t width,
               std::uint32_t height,
               float_type min_real,
               float_type max_real,
               float_type min_imag,
               float_type max_imag) {
  auto start_gpu = std::chrono::system_clock::now();
  auto clworker = spawn_cl<int*(float*)>(kernel, "mandelbrot",
                                         {width, height});
  vector<float_type> cljob;
  cljob.reserve(7);
  cljob.push_back(iterations);
  cljob.push_back(width);
  cljob.push_back(height);
  cljob.push_back(min_real);
  cljob.push_back(max_real);
  cljob.push_back(min_imag);
  cljob.push_back(max_imag);
#ifdef PRINT_IMAGE
  std::vector<QColor> palette;
  calculate_palette(palette, iterations);
#endif
  self->sync_send(clworker, std::move(cljob)).then (
    [=](const vector<int>& result) {
      static_cast<void>(result);
      DEBUG("Mandelbrot on GPU calculated");
#ifdef PRINT_IMAGE
      color_and_print(palette, result, width, height, "gpu");
#endif
      time_gpu = chrono::duration_cast<chrono::microseconds>(
        chrono::system_clock::now() - start_gpu
      ).count();
    }
  );
}

void usage(const char* name) {
  cout << "usage: ./" << name << " <%onCPU>" << endl
       << "   Argument specifies the percentage calculated on the CPU,"
          "   the rest is calculated on the GPU (default 100)" << endl;
  exit(0);
}

template<typename T>
T get_cut(T start, T end, uint32_t percentage) {
  auto dist = (abs(start) + abs(end)) * percentage / 100.0;
  return (dist - abs(start));
}

template<typename T>
T get_bottom(T distance, uint32_t percentage) {
  return distance * percentage / 100;
}

template<typename T>
T get_top(T distance, uint32_t percentage) {
  return distance * (100 - percentage) / 100;
}

int main(int argc, char** argv) {
  uint32_t on_cpu = 100;
  if (argc == 2) {
    on_cpu = stoul(argv[1]);
    if (on_cpu > 100) {
      usage(argv[0]);
    }
  } else if (argc != 1) {
    usage(argv[0]);
  }

  auto iterations = default_iterations;
  auto width      = default_width;
  auto height     = default_height;
  auto min_re     = default_min_real;
  auto max_re     = default_max_real;
  auto min_im     = default_min_imag;
  auto max_im     = default_max_imag;

  auto scale = [&](float_type ratio) {
    float_type abs_re = fabs(max_re + (-1 * min_re)) / 2;
    float_type abs_im = fabs(max_im + (-1 * min_im)) / 2;
    float_type mid_re = min_re + abs_re;
    float_type mid_im = min_im + abs_im;
    min_re = mid_re - (abs_re * ratio);
    max_re = mid_re + (abs_re * ratio);
    min_im = mid_im - (abs_im * ratio);
    max_im = mid_im + (abs_im * ratio); // min_im + (max_re - min_re) * height / width;
  };
  scale(0.3);

  auto cpu_width  = get_bottom(width, on_cpu);
  auto cpu_height = height;
  auto cpu_min_re = min_re;
  auto cpu_max_re = get_cut(min_re, max_re, on_cpu);
  auto cpu_min_im = min_im;
  auto cpu_max_im = max_im;

  auto gpu_width  = get_top(width, on_cpu);
  auto gpu_height = height;
  auto gpu_min_re = get_cut(min_re, max_re, on_cpu);
  auto gpu_max_re = max_re;
  auto gpu_min_im = min_im;
  auto gpu_max_im = max_im;

  DEBUG("[cpu] width: " << cpu_width
        << "(" << cpu_min_re << " to " << cpu_max_re << ")");
  DEBUG("[gpu] width: " << gpu_width
        << "(" << gpu_min_re << " to " << gpu_max_re << ")");

  auto kernel = caf::opencl::program::create(kernel_source);
  auto start = std::chrono::system_clock::now();
  if (gpu_width > 0) {
    // trigger calculation on the GPU
    spawn(mandel_cl, kernel, iterations, gpu_width, gpu_height,
          gpu_min_re, gpu_max_re, gpu_min_im, gpu_max_im);
  }
  auto start_cpu = std::chrono::system_clock::now();
  if (cpu_width > 0) {
    // trigger calculation on the CPU
    std::vector<int> image(cpu_width * cpu_height);
    auto re_factor = (cpu_max_re - cpu_min_re) / (cpu_width - 1);
    auto im_factor = (cpu_max_im - cpu_min_im) / (cpu_height - 1);
    for (uint32_t y = 0; y < cpu_height; ++y) {
      int* line = &image[y * cpu_width];
      spawn([=] {
        for (uint32_t x = 0; x < cpu_width; ++x) {
          auto z_re = cpu_min_re + x * re_factor;
          auto z_im = cpu_max_im - y * im_factor;
          auto const_re = z_re;
          auto const_im = z_im;
          uint32_t cnt = 0;
          float_type cond = 0;
          do {
            auto tmp_re = z_re;
            auto tmp_im = z_im;
            z_re = (tmp_re * tmp_re - tmp_im * tmp_im) + const_re;
            z_im = (2 * tmp_re * tmp_im) + const_im;
            cond = z_re * z_re + z_im * z_im;
            ++cnt;
          } while (cnt < iterations && cond <= 4.0f);
          line[x] = cnt;
        }
      });
    }
#ifdef PRINT_IMAGE
    std::vector<QColor> palette;
    calculate_palette(palette, iterations);
#endif
    await_all_actors_done();
    DEBUG("Mandelbrot on CPU calculated");
#ifdef PRINT_IMAGE
    color_and_print(palette, image, cpu_width, cpu_height, "cpu");
#endif
    time_cpu = chrono::duration_cast<chrono::microseconds>(
      chrono::system_clock::now() - start_cpu
    ).count();
  } else {
    await_all_actors_done();
  }
  shutdown();
  auto time_total = chrono::duration_cast<chrono::microseconds>(
    chrono::system_clock::now() - start
  ).count();
  cout << on_cpu << ", " << time_total
                 << ", " << time_cpu
                 << ", " << time_gpu
                 << endl;
  return 0;
}
