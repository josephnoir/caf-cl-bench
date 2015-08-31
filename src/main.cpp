
#include <chrono>

#ifdef PRINT_IMAGE
#include <QFile>
#include <QColor>
#include <QImage>
#include <QBuffer>
#include <QByteArray>
#endif

#include "config.hpp"

#include "caf/all.hpp"
#include "caf/opencl/spawn_cl.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

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

#ifdef CAF_DEBUG
#define DEBUG(x) cerr << x << endl;
#else
#define DEBUG(x)
#endif

} // namespace <anonymous>

// global values to track the time
unsigned long time_gpu = 0;
unsigned long time_cpu = 0;
unsigned long distribution = 100;

#ifdef PRINT_IMAGE
inline void calculate_palette(vector<QColor>& storage, uint32_t iterations) {
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

void color_and_print(const vector<QColor>& palette,
                     const vector<int>& counts,
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
  ostringstream fname;
  fname << "mandelbrot_" << distribution << "_" << identifier;
  fname << image_file_ending;
  QFile f{fname.str().c_str()};
  if (!f.open(QIODevice::WriteOnly)) {
    DEBUG("could not open file: " << fname.str());
  } else {
    img.save(&f, image_format);
    DEBUG(identifier << " image saved");
  }
}
#endif // PRINT_IMAGE

#ifdef ENABLE_GPU
// calculates mandelbrot that contains the iteration count on GPU
void mandel_cl(event_based_actor* self,
               program kernel,
               uint32_t iterations,
               uint32_t width,
               uint32_t height,
               float_type min_real,
               float_type max_real,
               float_type min_imag,
               float_type max_imag) {
  using namespace caf::opencl;
  auto unbox_args = [](message& msg) -> optional<message> {
    return msg;
  };
  auto box_res = [] (vector<int> result) -> message {
    return make_message(move(result), chrono::system_clock::now());
  };
  vector<float_type> cljob {
    static_cast<float_type>(iterations),
    static_cast<float_type>(width),
    static_cast<float_type>(height),
    min_real, max_real,
    min_imag, max_imag
  };
  spawn_config conf{dim_vec{width, height}};
  auto start_gpu = chrono::system_clock::now();
  auto clworker = spawn_cl(kernel, "mandelbrot", conf, unbox_args, box_res,
                           in<vector<float_type>>{}, out<vector<int>>{});
  self->sync_send(clworker, move(cljob)).then (
    [=](const vector<int>& result, const chrono::system_clock::time_point& end_gpu) {
      static_cast<void>(result);
      DEBUG("Mandelbrot on GPU calculated");
#ifdef PRINT_IMAGE
      vector<QColor> palette;
      calculate_palette(palette, iterations);
      color_and_print(palette, result, width, height, "gpu");
#endif // PRINT_IMAGE
      time_gpu = chrono::duration_cast<chrono::milliseconds>(
        end_gpu - start_gpu
      ).count();
    }
  );
}
#endif // ENABLE_GPU

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

  distribution = on_cpu;

  auto iterations = default_iterations;
  auto width      = default_width;
  auto height     = default_height;
  auto min_re     = default_min_real;
  auto max_re     = default_max_real;
  auto min_im     = default_min_imag;
  auto max_im     = default_max_imag;

  auto scale = [&](const float_type ratio) {
    float_type abs_re = fabs(max_re + (-1 * min_re)) / 2;
    float_type abs_im = fabs(max_im + (-1 * min_im)) / 2;
    float_type mid_re = min_re + abs_re;
    float_type mid_im = min_im + abs_im;
    auto dist = abs_re * ratio;
    min_re = mid_re - dist;
    max_re = mid_re + dist;
    min_im = mid_im - dist;
    max_im = mid_im + dist;
  };
  scale(default_scaling);

#ifdef ENABLE_CPU
  auto cpu_width  = get_bottom(width, on_cpu);
  auto cpu_height = height;
  auto cpu_min_re = min_re;
  auto cpu_max_re = get_cut(min_re, max_re, on_cpu);
  auto cpu_min_im = min_im;
  auto cpu_max_im = max_im;
  DEBUG("[CPU] width: " << cpu_width
        << "(" << cpu_min_re << " to " << cpu_max_re << ")");
#endif // ENABLE_CPU

#ifdef ENABLE_GPU
  auto gpu_width  = get_top(width, on_cpu);
  auto gpu_height = height;
  auto gpu_min_re = get_cut(min_re, max_re, on_cpu);
  auto gpu_max_re = max_re;
  auto gpu_min_im = min_im;
  auto gpu_max_im = max_im;
  DEBUG("[GPU] width: " << gpu_width
        << "(" << gpu_min_re << " to " << gpu_max_re << ")");
#endif // ENABLE_GPU

  auto kernel = program::create(kernel_source);
  auto start = chrono::system_clock::now();

#ifdef ENABLE_GPU
  if (gpu_width > 0) {
    // trigger calculation on the GPU
    spawn(mandel_cl, kernel, iterations, gpu_width, gpu_height,
          gpu_min_re, gpu_max_re, gpu_min_im, gpu_max_im);
  }
#endif // ENABLE_GPU

#ifdef ENABLE_CPU
  auto start_cpu = chrono::system_clock::now();
  if (cpu_width > 0) {
    // trigger calculation on the CPU
    vector<int> image(cpu_width * cpu_height);
    auto re_factor = (cpu_max_re - cpu_min_re) / (cpu_width - 1);
    auto im_factor = (cpu_max_im - cpu_min_im) / (cpu_height - 1);
    int* indirection = image.data();
    for (uint32_t im = 0; im < cpu_height; ++im) {
      spawn([=] {
        for (uint32_t re = 0; re < cpu_width; ++re) {
          auto z_re = cpu_min_re + re * re_factor;
          auto z_im = cpu_max_im - im * im_factor;
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
          indirection[re + im * cpu_width] = cnt;
        }
      });
    }
    await_all_actors_done();
#ifdef PRINT_IMAGE
    vector<QColor> palette;
    calculate_palette(palette, iterations);
    color_and_print(palette, image, cpu_width, cpu_height, "cpu");
#endif // PRINT_IMAGE
    DEBUG("Mandelbrot on CPU calculated");
    time_cpu = chrono::duration_cast<chrono::milliseconds>(
      chrono::system_clock::now() - start_cpu
    ).count();
  }
#endif

  await_all_actors_done();
  shutdown();
  auto time_total = chrono::duration_cast<chrono::milliseconds>(
    chrono::system_clock::now() - start
  ).count();
  cout << (100 - on_cpu)
       << ", " << time_total
       << ", " << time_cpu
       << ", " << time_gpu
       << endl;
  return 0;
}
