// 컴파일러 탐침: 컴파일 시점에 값이 정해지는 피연산자로 구간 연산을 하고 끝점을 16진(%a)으로 출력한다.
// 판정은 하지 않는다 — tests/test_compiler_probe.py 가 Fraction 으로 정확값과 대조한다.
#include <cstdio>
#include "capd/capdlib.h"

#ifdef T3CAP_NO_ROUNDING
// 음성 대조: 같은 CAPD 구간 템플릿에서 반올림 전환만 없앤다. 검사기는 이 출력에서 반드시 실패해야 한다
struct NoRounding {
  static void roundNearest() {}
  static void roundUp() {}
  static void roundDown() {}
};
using interval = capd::intervals::Interval<double, NoRounding>;
using IVector = capd::vectalg::Vector<interval, 0>;
using IMatrix = capd::vectalg::Matrix<interval, 0, 0>;
#else
using capd::interval;
using capd::IVector;
using capd::IMatrix;
#endif

namespace {

void emit(const char* name, const interval& r) { std::printf("%s %a %a\n", name, r.leftBound(), r.rightBound()); }

#ifndef T3CAP_NO_ROUNDING
// 엔진 벡터장과 같이 상수 노드를 쓴 식을 libcapd 에 컴파일된 자동미분 경로로 계산한다(x=1 에서 평가)
void constantField(capd::autodiff::Node, capd::autodiff::Node in[], int, capd::autodiff::Node out[], int,
                   capd::autodiff::Node[], int) {
  out[0] = in[0] / 3.0;
  out[1] = in[0] * 0.1 + 0.2;
  out[2] = sqrt(in[0] + in[0]);
  out[3] = (in[0] - 1e-20) * 0.7;
}
#endif

}  // namespace

int main() {
  emit("div_1_3", interval(1.0) / interval(3.0));
  emit("div_1_10", interval(1.0) / interval(10.0));
  emit("add_01_02", interval(0.1) + interval(0.2));
  emit("sub_1_1e-20", interval(1.0) - interval(1e-20));
  emit("mul_01_3", interval(0.1) * interval(3.0));
  emit("mul_01_07", interval(0.1) * interval(0.7));
  emit("sqr_01", sqr(interval(0.1)));
  emit("sqrt_2", sqrt(interval(2.0)));
  emit("muladd_01_07_03", interval(0.1) * interval(0.7) + interval(0.3));
  IMatrix m(2, 2);
  m[0][0] = interval(0.1);
  m[0][1] = interval(0.2);
  m[1][0] = interval(0.3);
  m[1][1] = interval(0.7);
  IVector v(2);
  v[0] = interval(3.0);
  v[1] = interval(0.1);
  const IVector w = m * v;
  emit("matvec_0", w[0]);
  emit("matvec_1", w[1]);
#ifndef T3CAP_NO_ROUNDING
  capd::IMap f(constantField, 1, 4, 0);
  IVector x(1);
  x[0] = interval(1.0);
  const IVector y = f(x);
  emit("map_div_3", y[0]);
  emit("map_muladd_01_02", y[1]);
  emit("map_sqrt_2", y[2]);
  emit("map_sub_mul_07", y[3]);
#endif
  return 0;
}
