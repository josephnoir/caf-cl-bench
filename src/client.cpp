#include <map>
#include <chrono>
#include <time.h>
#include <vector>
#include <cstdlib>
#include <iostream>

#include <QImage>
#include <QColor>
#include <QBuffer>
#include <QByteArray>

#include "caf/all.hpp"

#include "client.hpp"
#include "config.hpp"
#include "calculate_fractal.hpp"
#include "q_byte_array_info.hpp"

#include "caf/opencl/spawn_cl.hpp"

using namespace std;
using namespace caf;

message response_from_image(uint32_t image_id, QImage image) {
  QByteArray ba;
  QBuffer buf{&ba};
  buf.open(QIODevice::WriteOnly);
  image.save(&buf, image_format);
  buf.close();
  return make_message(atom("Result"), image_id, ba);
}

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

using palette_ptr = shared_ptr<vector<QColor>>;

class clbroker : public event_based_actor {
 public:
  behavior make_behavior() {
    return {
      [=](mandel_atom, const actor& last_sender,
          uint32_t width, uint32_t height,
          uint32_t iterations, uint32_t image_id,
          float_type min_re, float_type max_re,
          float_type min_im, float_type max_im) {
          m_current_server = last_sender;
          if (   width != clwidth
              || height != clheight
              || iterations != cliterations) {
              calculate_palette(palette, iterations);
              clworker = spawn_cl<int*(float*)>(clprog, "mandel",
                                                {width, height});
              clwidth = width;
              clheight = height;
              cliterations = iterations;
          }
          clforward(image_id, min_re, max_re, min_im, max_im);
      },
      [=] (ident_atom) {
          return make_tuple(atom("opencl"), this);
      },
      others() >> [=]{
          aout(this) << "Unexpected message: '"
               << to_string(last_dequeued()) << "'.\n";
      }
    };
  }

  clbroker(uint32_t device_id)
      : clprog(opencl::program::create(kernel_source, nullptr, device_id)),
        clwidth(0),
        clheight(0),
        cliterations(0) {
    // nop
  }

 private:
  void clforward(uint32_t image_id,
                 float_type min_re, float_type max_re,
                 float_type min_im, float_type max_im) {
    m_current_server = last_sender();
    vector<float> cljob;
    cljob.reserve(7);
    cljob.push_back(cliterations);
    cljob.push_back(clwidth);
    cljob.push_back(clheight);
    cljob.push_back(min_re);
    cljob.push_back(max_re);
    cljob.push_back(min_im);
    cljob.push_back(max_im);
    auto hdl = make_response_handle();
    sync_send(clworker, std::move(cljob)).then (
      [=](const vector<int>& result) {
        QImage image{static_cast<int>(clwidth),
                     static_cast<int>(clheight),
                     QImage::Format_RGB32};
        for (uint32_t y = 0; y < clheight; ++y) {
          for (uint32_t x = 0; x < clwidth; ++x) {
            // double n = result[x+y*clwidth];
            // double n_min = 1;
            // double n_max = cliterations;
            // auto u = log(n/n_min) / log(n_max/n_min);
            // uint32_t idx = u * cliterations;
            // image.setPixel(x,y,palette[idx].rgb());
            image.setPixel(x,y,palette[result[x+y*clwidth]].rgb());
          }
        }
        reply_tuple_to(hdl, response_from_image(image_id, image));
      }
    );
  }

  opencl::program clprog;
  uint32_t clwidth;
  uint32_t clheight;
  uint32_t cliterations;
  actor clworker;
  actor m_current_server;
  std::vector<QColor> palette;
};

actor spawn_opencl_client(uint32_t device_id) {
   return spawn<clbroker>(device_id);
}

behavior client::make_behavior(){
  return {
    [=](mandel_atom, actor server,
        uint32_t width,uint32_t height,
        uint32_t iterations, uint32_t image_id,
        float_type min_re, float_type max_re,
        float_type min_im, float_type max_im) {
      bool frac_changed = false;
      if (m_current_fractal_type != atom("mandel")) {
        m_current_fractal_type  = atom("mandel");
        frac_changed = true;
      }
      send(
        server,
        response_from_image(
          image_id,
          calculate_mandelbrot(
            m_palette, width,
            height, iterations,
            min_re, max_re,
            min_im, max_im,
            frac_changed
          )
        )
      );
    },
    [=] (ident_atom) {
      return make_message(atom("normal"), this);
    },
    others() >> [=] {
      aout(this) << "Client received unexpected message: "
                 << to_string(last_dequeued()) << endl;
    }
  };
}
