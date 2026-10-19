// 구간 산술 자체 시험: 무작위 double 로 구간 연산을 하고 끝점을 16진(%a)으로 출력한다.
// 판정은 하지 않는다 — tests/test_arith_selftest.py 가 Fraction 으로 정확값과 대조한다.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "capd/capdlib.h"

#ifdef T3CAP_NO_ROUNDING
// 음성 대조(G6): 같은 CAPD 구간 템플릿에서 반올림 전환만 없앤다. 검사기는 이 출력에서 반드시 실패해야 한다
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

uint64_t state = 0x243F6A8885A308D3ull;

uint64_t nextBits() {  // splitmix64 — 결정적 재현
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double randomDouble() {
  double mantissa = static_cast<double>(nextBits() >> 11) * 0x1.0p-53 + 0.5;
  int exponent = static_cast<int>(nextBits() % 41) - 20;
  double v = std::ldexp(mantissa, exponent);
  return (nextBits() & 1u) ? -v : v;
}

void emit(const char* op, double a, double b, const interval& r) {
  std::printf("%s %a %a %a %a\n", op, a, b, r.leftBound(), r.rightBound());
}

void emit4(const char* op, double a1, double a2, double b1, double b2, const interval& r) {
  std::printf("%s %a %a %a %a %a %a\n", op, a1, a2, b1, b2, r.leftBound(), r.rightBound());
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
  for (int k = 0; k < n; ++k) {
    const double a = randomDouble(), b = randomDouble();
    const interval ia(a), ib(b);
    emit("add", a, b, ia + ib);
    emit("sub", a, b, ia - ib);
    emit("mul", a, b, ia * ib);
    emit("div", a, b, ia / ib);
    emit("sqrt", std::fabs(a), 0.0, sqrt(interval(std::fabs(a))));

    double a1 = randomDouble(), a2 = randomDouble(), b1 = randomDouble(), b2 = randomDouble();
    if (a1 > a2) std::swap(a1, a2);
    if (b1 > b2) std::swap(b1, b2);
    const interval A(a1, a2), B(b1, b2);
    emit4("iadd", a1, a2, b1, b2, A + B);
    emit4("isub", a1, a2, b1, b2, A - B);
    emit4("imul", a1, a2, b1, b2, A * B);
    if (b1 > 0 || b2 < 0) emit4("idiv", a1, a2, b1, b2, A / B);
    const double s1 = std::fabs(a1), s2 = std::fabs(a2);
    emit4("isqrt", std::fmin(s1, s2), std::fmax(s1, s2), 0.0, 0.0, sqrt(interval(std::fmin(s1, s2), std::fmax(s1, s2))));
  }
  // 인라인된 행렬·벡터 경로: 반올림 전환이 최적화로 사라지지 않았는지 본다
  for (int k = 0; k < n / 10; ++k) {
    IMatrix M(3, 3);
    IVector v(3);
    double md[9], vd[3];
    for (int i = 0; i < 9; ++i) { md[i] = randomDouble(); M[i / 3][i % 3] = interval(md[i]); }
    for (int i = 0; i < 3; ++i) { vd[i] = randomDouble(); v[i] = interval(vd[i]); }
    const IVector w = M * v;
    for (int i = 0; i < 3; ++i) {
      std::printf("matvec");
      for (int j = 0; j < 3; ++j) std::printf(" %a %a", md[3 * i + j], vd[j]);
      std::printf(" %a %a\n", w[i].leftBound(), w[i].rightBound());
    }
  }
  return 0;
}
