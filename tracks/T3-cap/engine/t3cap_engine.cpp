// 회전을 제거한 축소 좌표 평면 3체 계의 Poincaré 사상 C0·C1 구간 감싸기 엔진(CAPD).
// 표준입력의 토큰 요청을 읽어 표준출력에 상자별 결과를 쓴다. 판정은 하지 않는다 — t3cap/verdict.py 소관.
// 좌표·프로토콜 정의: docs/T3-cap/REPORT.md 「축소 좌표」「엔진 프로토콜」.
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"
#ifdef __HAVE_MPFR__
#include "capd/mpcapdlib.h"
#include "capd/dynsys/MpStepControl.h"
// MP 단계 제어 특수화는 라이브러리에 인스턴스가 없어 구현 헤더로 이 번역 단위에서 만든다(CAPD mpExample5 와 같은 방식)
#include "capd/dynsys/OdeSolver.hpp"
#include "capd/dynsys/BasicOdeSolver.hpp"
#include "capd/poincare/PoincareMap.hpp"
#include "capd/poincare/BasicPoincareMap.hpp"
#endif

using capd::autodiff::Node;

namespace t3cap {

std::string firstLine(const std::string& s);

// 어느 단계에서 실패했는지 메시지에 남긴다(관측용 — 실패는 그대로 실패로 전달된다)
template <class F>
auto stage(const char* name, F&& body) -> decltype(body()) {
  try {
    return body();
  } catch (std::exception& e) {
    throw std::runtime_error(std::string(name) + ": " + e.what());
  }
}

constexpr int kPhaseDim = 6;  // y = (r, x, y, pr, px, py)
constexpr int kAngleIndex = 6;  // 회전각 theta 를 함께 적분할 때의 7번째 성분

// params = (m1, m2, m3, J). dimOut == 7 이면 회전각 속도 theta' 을 덧붙인다.
void reducedVectorField(Node, Node in[], int, Node out[], int dimOut, Node params[], int) {
  Node r = in[0], x = in[1], y = in[2], pr = in[3], px = in[4], py = in[5];
  Node m1 = params[0], m2 = params[1], m3 = params[2], J = params[3];
  Node m12 = m1 + m2;
  Node a = m2 / m12;
  Node b = m1 / m12;
  Node mu1 = m1 * m2 / m12;
  Node mu2 = m12 * m3 / (m12 + m3);
  Node L = J - x * py + y * px;
  Node omega = L / (mu1 * r * r);
  Node d13 = x + a * r;
  Node d23 = x - b * r;
  Node s13 = d13 * d13 + y * y;
  Node s23 = d23 * d23 + y * y;
  // 엄밀 경로에 초월함수를 넣지 않으려고 r^-3 을 sqrt 와 나눗셈으로만 만든다
  Node g13 = m1 * m3 / (s13 * sqrt(s13));
  Node g23 = m2 * m3 / (s23 * sqrt(s23));
  out[0] = pr / mu1;
  out[1] = px / mu2 + y * omega;
  out[2] = py / mu2 - x * omega;
  out[3] = L * omega / r - m1 * m2 / (r * r) - a * g13 * d13 + b * g23 * d23;
  out[4] = py * omega - g13 * d13 - g23 * d23;
  out[5] = -px * omega - (g13 + g23) * y;
  if (dimOut == 7) out[6] = omega;
}

// ---------------------------------------------------------------- 16진 끝점 해석

// 16진 토큰의 꼴([+-]0x 자릿수[.자릿수][p[+-]10진 지수], 64자 이하)과, 값을 홀수 × 2^lowest 로 쓸 때의 홀수 비트 수
struct HexToken {
  bool negative = false;
  std::string significant;  // 앞뒤 0 을 뗀 16진 자릿수(값이 0 이면 비어 있다)
  long long scale = 0;      // significant 의 마지막 자릿수가 나타내는 2진 지수
  long long oddBits = 0;
  long long lowest = 0;
};

HexToken parseHexToken(const std::string& tok) {
  const auto bad = [&tok] { return std::runtime_error("request: bad hex float: " + tok); };
  const size_t sign = (!tok.empty() && (tok[0] == '+' || tok[0] == '-')) ? 1 : 0;
  if (tok.size() > 64 || (tok.compare(sign, 2, "0x") != 0 && tok.compare(sign, 2, "0X") != 0)) throw bad();
  std::string digits;
  long long fracDigits = 0, exponent = 0;
  bool point = false;
  size_t i = sign + 2;
  for (; i < tok.size() && tok[i] != 'p' && tok[i] != 'P'; ++i) {
    if (tok[i] == '.' && !point) { point = true; continue; }
    if (!std::isxdigit(static_cast<unsigned char>(tok[i]))) throw bad();
    digits += tok[i];
    fracDigits += point ? 1 : 0;
  }
  if (digits.empty()) throw bad();
  if (i < tok.size()) {
    const std::string field = tok.substr(i + 1);
    const size_t fieldSign = (!field.empty() && (field[0] == '+' || field[0] == '-')) ? 1 : 0;
    if (field.size() == fieldSign || field.size() > 8 || field.find_first_not_of("0123456789", fieldSign) != std::string::npos)
      throw bad();
    exponent = std::stoll(field);
  }
  HexToken h;
  h.negative = tok[0] == '-';
  const size_t first = digits.find_first_not_of('0');
  if (first == std::string::npos) return h;
  const size_t last = digits.find_last_not_of('0');
  h.significant = digits.substr(first, last - first + 1);
  h.scale = exponent - 4 * fracDigits + 4 * static_cast<long long>(digits.size() - 1 - last);
  const int lead = std::stoi(h.significant.substr(0, 1), nullptr, 16);
  const int tail = std::stoi(h.significant.substr(h.significant.size() - 1), nullptr, 16);
  const int leadBits = lead >= 8 ? 4 : lead >= 4 ? 3 : lead >= 2 ? 2 : 1;
  const int tailZeros = (tail & 1) ? 0 : (tail & 2) ? 1 : (tail & 4) ? 2 : 3;
  h.oddBits = 4 * static_cast<long long>(h.significant.size() - 1) + leadBits - tailZeros;
  h.lowest = h.scale + tailZeros;
  return h;
}

// double 실행의 16진 끝점: 값이 double 로 정확히 나타날 때만 받는다 — strtod 는 넘치는 자릿수를 최근접으로 반올림해 점 구간이 참값을 놓친다
double exactHexDouble(const std::string& tok) {
  const HexToken h = parseHexToken(tok);
  double magnitude = 0;
  if (!h.significant.empty()) {
    // double 은 홀수 가수 53비트 이하, 최하위 비트 지수 −1074 이상, 최상위 비트 지수 1023 이하인 수다
    if (h.oddBits > 53 || h.lowest < -1074 || h.lowest + h.oddBits - 1 > 1023)
      throw std::runtime_error("request: hex endpoint is not exactly a double: " + tok);
    magnitude = std::ldexp(static_cast<double>(std::stoull(h.significant, nullptr, 16)), static_cast<int>(h.scale));
  }
  char* end = nullptr;
  const double v = std::strtod(tok.c_str(), &end);
  if (end == tok.c_str() || *end != '\0' || v != (h.negative ? -magnitude : magnitude))
    throw std::runtime_error("request: bad hex float: " + tok);
  return v;
}

struct DoubleArith {
  using Interval = capd::interval;
  using IVector = capd::IVector;
  using IMatrix = capd::IMatrix;
  using IMap = capd::IMap;
  using Solver = capd::IOdeSolver;
  using PoincareMap = capd::IPoincareMap;
  using Section = capd::ICoordinateSection;
  using C0Set = capd::C0Rect2Set;
  using C1Set = capd::C1Rect2Set;

  static void configure(int) {}
  static std::string name() { return "double"; }
  static Interval fromDouble(double v) { return Interval(v); }
  static Interval fromHex(const std::string& tok) { return Interval(exactHexDouble(tok)); }
  static Interval fromInteger(long long v) { return Interval(static_cast<double>(v)); }
  static Interval hull(const Interval& lo, const Interval& hi) { return Interval(lo.leftBound(), hi.rightBound()); }
  // (lo+hi)/2 는 어떤 반올림 모드에서도 [lo,hi] 안에 든다(단조성) — 평균값 형식의 중심 조건
  static Interval thin(const Interval& v) { return Interval((v.leftBound() + v.rightBound()) / 2); }
  static std::string fmt(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", v);
    return buf;
  }
  static std::string fmtLo(const Interval& v) { return fmt(v.leftBound()); }
  static std::string fmtHi(const Interval& v) { return fmt(v.rightBound()); }
};

#ifdef __HAVE_MPFR__
// MpReal 은 mpfr 표현을 protected 로 두므로 16진 해석에만 파생해 쓴다(정밀도 = MpArith::configure 가 정한 기본 정밀도)
struct MpfrHex : capd::MpFloat {
  // MPFR 삼항 값: 0 이면 반올림 없이 읽었다. 토큰 끝까지 읽지 못하면 2
  int read(const std::string& tok) {
    char* end = nullptr;
    const int ternary = mpfr_strtofr(mpfr_rep, tok.c_str(), &end, 16, MPFR_RNDN);
    return *end == '\0' ? ternary : 2;
  }
};

struct MpArith {
  using Interval = capd::MpInterval;
  using IVector = capd::MpIVector;
  using IMatrix = capd::MpIMatrix;
  using IMap = capd::MpIMap;
  using Solver = capd::dynsys::OdeSolver<capd::MpIMap, capd::dynsys::MpLastTermsStepControl>;
  using PoincareMap = capd::poincare::PoincareMap<Solver>;
  using Section = capd::MpICoordinateSection;
  using C0Set = capd::MpC0Rect2Set;
  using C1Set = capd::MpC1Rect2Set;

  static int& bits() { static int b = 128; return b; }
  static void configure(int precisionBits) {
    // 16진 double 과 |p|,q ≤ 2^53 인 p/q 끝점을 반올림 없이 담으려면 가수가 53비트 이상이어야 한다
    if (precisionBits < 53)
      throw std::runtime_error("request: mp precision must be at least 53 bits, got " + std::to_string(precisionBits));
    bits() = precisionBits;
    capd::MpFloat::setDefaultPrecision(precisionBits);
  }
  static std::string name() { return "mp:" + std::to_string(bits()); }
  static Interval fromDouble(double v) { return Interval(capd::MpFloat(v)); }
  // MP 실행의 16진 끝점: 값이 설정 정밀도로 정확히 나타날 때만 받는다. MPFR 이 반올림을 알리면 두 판정이 어긋난 것이라 거부한다
  static Interval fromHex(const std::string& tok) {
    const HexToken h = parseHexToken(tok);
    if (h.oddBits > bits())
      throw std::runtime_error("request: hex endpoint is not exactly an mp:" + std::to_string(bits()) + " number: " + tok);
    MpfrHex v;
    if (v.read(tok) != 0) throw std::runtime_error("request: bad hex float: " + tok);
    return Interval(static_cast<const capd::MpFloat&>(v));
  }
  static Interval fromInteger(long long v) { return Interval(capd::MpFloat(static_cast<long>(v))); }
  static Interval hull(const Interval& lo, const Interval& hi) { return Interval(lo.leftBound(), hi.rightBound()); }
  static Interval thin(const Interval& v) {
    capd::MpFloat m = (v.leftBound() + v.rightBound()) / 2;
    return Interval(m);
  }
  static std::string fmtBound(const capd::MpFloat& v, capd::MpFloat::RoundingMode rnd) {
    std::ostringstream out;
    // 10진 출력 자체를 바깥쪽으로 반올림해 끝점이 원래 구간을 덮게 한다
    v.put(out, rnd, bits(), 10, static_cast<int>(bits() * 0.30103) + 8);
    return out.str();
  }
  static std::string fmtLo(const Interval& v) { return fmtBound(v.leftBound(), capd::MpFloat::RoundDown); }
  static std::string fmtHi(const Interval& v) { return fmtBound(v.rightBound(), capd::MpFloat::RoundUp); }
};
#endif

// ---------------------------------------------------------------- 요청 파싱

struct Tokens {
  std::vector<std::string> items;
  size_t pos = 0;
  bool done() const { return pos >= items.size(); }
  const std::string& next() {
    if (done()) throw std::runtime_error("request: unexpected end of input");
    return items[pos++];
  }
  int nextInt() { return std::stoi(next()); }
};

// 끝점 토큰은 16진 double("0x...", 정확) 또는 정수 유리수 "p/q" 만 받는다 — 10진 소수(0.1)는 함정 ① 이라 거부한다
template <class A>
typename A::Interval parseEndpoint(const std::string& tok) {
  const auto slash = tok.find('/');
  if (slash != std::string::npos) {
    const long long p = std::stoll(tok.substr(0, slash));
    const long long q = std::stoll(tok.substr(slash + 1));
    const long long limit = 1LL << 53;
    if (q == 0 || p > limit || p < -limit || q > limit || q < -limit)
      throw std::runtime_error("request: rational endpoint out of exact range: " + tok);
    return A::fromInteger(p) / A::fromInteger(q);
  }
  const bool hex = tok.find("0x") != std::string::npos || tok.find("0X") != std::string::npos;
  if (!hex) throw std::runtime_error("request: endpoint must be hex float or p/q rational: " + tok);
  return A::fromHex(tok);
}

template <class A>
typename A::Interval parseInterval(Tokens& t) {
  const auto lo = parseEndpoint<A>(t.next());
  const auto hi = parseEndpoint<A>(t.next());
  if (lo.leftBound() > hi.rightBound()) throw std::runtime_error("request: interval with lo > hi");
  return A::hull(lo, hi);
}

struct SectionSpec {
  int k = 2;       // 단면 좌표 성분 y_k = c
  int dir = 1;     // +1: s 가 증가하며 통과(MinusPlus), -1: 감소하며 통과(PlusMinus)
  int m = 5;       // 에너지 식으로 푸는 운동량 성분 (3,4,5 중 하나)
  int branch = 1;  // dH/dy_m 의 부호 = 근 선택
  std::vector<int> freeIdx;  // 차트 좌표 u 가 되는 나머지 4성분(오름차순)
};

SectionSpec parseSection(Tokens& t) {
  SectionSpec s;
  s.k = t.nextInt();
  s.dir = t.nextInt();
  s.m = t.nextInt();
  s.branch = t.nextInt();
  if (s.k < 0 || s.k >= kPhaseDim || s.m < 3 || s.m > 5 || s.m == s.k)
    throw std::runtime_error("request: invalid section indices");
  if ((s.dir != 1 && s.dir != -1) || (s.branch != 1 && s.branch != -1))
    throw std::runtime_error("request: dir and branch must be +1 or -1");
  for (int i = 0; i < kPhaseDim; ++i)
    if (i != s.k && i != s.m) s.freeIdx.push_back(i);
  return s;
}

template <class A>
struct Request {
  using Interval = typename A::Interval;
  using IVector = typename A::IVector;
  using IMatrix = typename A::IMatrix;

  std::string mode;
  int order = 20;
  bool fixedStep = false;
  Interval step;
  Interval m1, m2, m3, J, h;
  SectionSpec in, out;
  Interval cIn, cOut;  // 단면 값 y_k = c
  int returns = 1;
  bool rotation = false;
  IVector frameInCenter = IVector(4), frameOutCenter = IVector(4);
  IMatrix frameInMatrix = IMatrix::Identity(4), frameOutInverse = IMatrix::Identity(4);
  std::vector<IVector> boxes;
};

template <class A>
Request<A> parseRequest(Tokens& t) {
  Request<A> r;
  while (!t.done()) {
    const std::string key = t.next();
    if (key == "mode") r.mode = t.next();
    else if (key == "order") r.order = t.nextInt();
    else if (key == "step") {
      const std::string kind = t.next();
      if (kind == "fixed") { r.fixedStep = true; r.step = A::thin(parseEndpoint<A>(t.next())); }
      else if (kind != "adaptive") throw std::runtime_error("request: step must be adaptive|fixed");
    } else if (key == "masses") { r.m1 = parseInterval<A>(t); r.m2 = parseInterval<A>(t); r.m3 = parseInterval<A>(t); }
    else if (key == "energy") r.h = parseInterval<A>(t);
    else if (key == "angmom") r.J = parseInterval<A>(t);
    else if (key == "section_in") { r.in = parseSection(t); r.cIn = parseInterval<A>(t); }
    else if (key == "section_out") { r.out = parseSection(t); r.cOut = parseInterval<A>(t); }
    else if (key == "returns") r.returns = t.nextInt();
    else if (key == "rotation") r.rotation = t.nextInt() != 0;
    else if (key == "frame_in") {
      for (int i = 0; i < 4; ++i) r.frameInCenter[i] = parseInterval<A>(t);
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.frameInMatrix[i][j] = parseInterval<A>(t);
    } else if (key == "frame_out") {
      for (int i = 0; i < 4; ++i) r.frameOutCenter[i] = parseInterval<A>(t);
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.frameOutInverse[i][j] = parseInterval<A>(t);
    } else if (key == "boxes") {
      const int n = t.nextInt();
      const int dim = t.nextInt();
      for (int b = 0; b < n; ++b) {
        typename A::IVector v(dim);
        for (int i = 0; i < dim; ++i) v[i] = parseInterval<A>(t);
        r.boxes.push_back(v);
      }
    } else if (key == "end") break;
    else throw std::runtime_error("request: unknown key " + key);
  }
  if (r.returns < 1) throw std::runtime_error("request: returns must be >= 1");
  return r;
}

// ---------------------------------------------------------------- 계(질량·J)와 차트

template <class A>
class ReducedSystem {
 public:
  using Interval = typename A::Interval;
  using IVector = typename A::IVector;
  using IMatrix = typename A::IMatrix;
  using IMap = typename A::IMap;

  ReducedSystem(const Interval& m1, const Interval& m2, const Interval& m3, const Interval& J, const Interval& h)
      : m1_(m1), m2_(m2), m3_(m3), J_(J), h_(h) {
    const Interval m12 = m1 + m2;
    a_ = m2 / m12;
    b_ = m1 / m12;
    mu1_ = m1 * m2 / m12;
    mu2_ = m12 * m3 / (m12 + m3);
  }

  IMap vectorField(int dim) const {
    IMap f(reducedVectorField, dim, dim, 4);
    f.setParameter(0, m1_);
    f.setParameter(1, m2_);
    f.setParameter(2, m3_);
    f.setParameter(3, J_);
    return f;
  }

  struct Distances { Interval r12, r13, r23; };

  Distances distances(const IVector& y) const {
    Distances d;
    d.r12 = y[0];
    d.r13 = sqrt(sqr(y[1] + a_ * y[0]) + sqr(y[2]));
    d.r23 = sqrt(sqr(y[1] - b_ * y[0]) + sqr(y[2]));
    return d;
  }

  Interval potential(const IVector& y) const {
    const Distances d = distances(y);
    return m1_ * m2_ / d.r12 + m1_ * m3_ / d.r13 + m2_ * m3_ / d.r23;
  }

  Interval angularMomentumOfPair(const IVector& y) const { return J_ - y[1] * y[5] + y[2] * y[4]; }

  Interval hamiltonian(const IVector& y) const {
    const Interval L = angularMomentumOfPair(y);
    const Interval two = A::fromInteger(2);
    return (sqr(y[3]) + sqr(L) / sqr(y[0])) / (two * mu1_) + (sqr(y[4]) + sqr(y[5])) / (two * mu2_) -
           potential(y);
  }

  // H 를 y_m 의 2차식 qa y_m^2 + qb y_m + qc 로 쓴 계수 (y_m 자리는 무시)
  void quadraticInMomentum(const IVector& y, int m, Interval& qa, Interval& qb, Interval& qc) const {
    const Interval one = A::fromInteger(1), two = A::fromInteger(2);
    const Interval inv2mu1r2 = one / (two * mu1_ * sqr(y[0]));
    const Interval U = potential(y);
    if (m == 5) {
      const Interval base = J_ + y[2] * y[4];
      qa = sqr(y[1]) * inv2mu1r2 + one / (two * mu2_);
      qb = -two * base * y[1] * inv2mu1r2;
      qc = sqr(y[3]) / (two * mu1_) + sqr(base) * inv2mu1r2 + sqr(y[4]) / (two * mu2_) - U;
    } else if (m == 4) {
      const Interval base = J_ - y[1] * y[5];
      qa = sqr(y[2]) * inv2mu1r2 + one / (two * mu2_);
      qb = two * base * y[2] * inv2mu1r2;
      qc = sqr(y[3]) / (two * mu1_) + sqr(base) * inv2mu1r2 + sqr(y[5]) / (two * mu2_) - U;
    } else {
      const Interval L = angularMomentumOfPair(y);
      qa = one / (two * mu1_);
      qb = A::fromInteger(0);
      qc = sqr(L) * inv2mu1r2 + (sqr(y[4]) + sqr(y[5])) / (two * mu2_) - U;
    }
  }

  // 차트 Phi: 단면 위 자유 좌표 u(4) -> y(6). D(판별식) 를 함께 돌려준다(함정 ⑤: 호출자가 D>0 확인)
  IVector chart(const IVector& u, const SectionSpec& s, const Interval& c, Interval& D) const {
    IVector y(kPhaseDim);
    y[s.k] = c;
    for (int i = 0; i < 4; ++i) y[s.freeIdx[i]] = u[i];
    y[s.m] = A::fromInteger(0);
    Interval qa, qb, qc;
    quadraticInMomentum(y, s.m, qa, qb, qc);
    D = sqr(qb) - A::fromInteger(4) * qa * (qc - h_);
    if (!(D.leftBound() > 0)) throw std::runtime_error("chart: discriminant not proven positive (trap 5)");
    y[s.m] = (-qb + A::fromInteger(s.branch) * sqrt(D)) / (A::fromInteger(2) * qa);
    return y;
  }

  // dH/dy = (-f3, -f4, -f5, f0, f1, f2): 해밀턴 방정식에서 기울기를 벡터장으로 얻는다
  IVector gradientH(IMap& f, const IVector& y) const {
    IVector y6(kPhaseDim);
    for (int i = 0; i < kPhaseDim; ++i) y6[i] = y[i];
    const IVector fy = f(y6);
    IVector g(kPhaseDim);
    for (int i = 0; i < 3; ++i) { g[i] = -fy[i + 3]; g[i + 3] = fy[i]; }
    return g;
  }

  // DPhi(U) (6x4): 자유 성분은 단위행, 단면 성분 0, y_m 행은 음함수 미분 -dH/du / (branch*sqrt(D))
  IMatrix chartDerivative(IMap& f6, const IVector& yBox, const SectionSpec& s, const Interval& D) const {
    IMatrix d(kPhaseDim, 4);
    const IVector g = gradientH(f6, yBox);
    const Interval dHdm = A::fromInteger(s.branch) * sqrt(D);
    for (int i = 0; i < 4; ++i) {
      d[s.freeIdx[i]][i] = A::fromInteger(1);
      d[s.m][i] = -g[s.freeIdx[i]] / dHdm;
    }
    return d;
  }

  const Interval& a() const { return a_; }
  const Interval& b() const { return b_; }

 private:
  Interval m1_, m2_, m3_, J_, h_;
  Interval a_, b_, mu1_, mu2_;
};

// ---------------------------------------------------------------- 출력

template <class A>
struct Writer {
  std::ostringstream out;
  void scalar(const char* key, const typename A::Interval& v) {
    out << key << ' ' << A::fmtLo(v) << ' ' << A::fmtHi(v) << '\n';
  }
  void vector(const char* key, const typename A::IVector& v, int n) {
    out << key;
    for (int i = 0; i < n; ++i) out << ' ' << A::fmtLo(v[i]) << ' ' << A::fmtHi(v[i]);
    out << '\n';
  }
  void matrix(const char* key, const typename A::IMatrix& m, int rows, int cols) {
    out << key << ' ' << rows << ' ' << cols;
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) out << ' ' << A::fmtLo(m[i][j]) << ' ' << A::fmtHi(m[i][j]);
    out << '\n';
  }
};

// ---------------------------------------------------------------- 출력 단면 통과 세기(함정 ②)

// CAPD PoincareMap 은 집합이 출력 단면을 떠나는 동안의 통과를 세지 않으므로(PoincareMap_templateMembers.h),
// 흐름 관 단계로 통과를 따로 세어 CAPD 가 낸 반환이 t>0 의 returns 번째 지정 방향 통과인지 확인한다.
// s 가 닿는 연속 단계 묶음(run)마다 ds/dt 부호가 한결같아야 한다 — 엄격 단조라 묶음 안 영점은 많아야 하나다.
template <class A>
class CrossingCounter {
 public:
  using Interval = typename A::Interval;

  // initialSign: 초기 집합 위 s 의 부호(0 = 단면에 걸침). onSection: 초기 집합이 출력 단면 위(s≡0)라 첫 묶음의 영점은 t=0
  CrossingCounter(int dir, int returns, const Interval& window, int initialSign, bool onSection)
      : dir_(dir), returns_(returns), window_(window), before_(initialSign), onSection_(onSection) {}

  void step(const Interval& s, const Interval& dsdt, const Interval& t0, const Interval& t1) {
    if (s.leftBound() > 0 || s.rightBound() < 0) {
      const int sign = s.leftBound() > 0 ? 1 : -1;
      if (runSign_ != 0) closeRun(sign);
      before_ = sign;
      first_ = false;
      return;
    }
    if (!(dsdt.leftBound() > 0 || dsdt.rightBound() < 0))
      throw std::runtime_error("crossing: output section touched where ds/dt may vanish (trap 2)");
    const int sign = dsdt.leftBound() > 0 ? 1 : -1;
    if (runSign_ == 0) {
      runSign_ = sign;
      runFromStart_ = first_;
      runStart_ = t0;
    } else if (sign != runSign_) {
      throw std::runtime_error("crossing: ds/dt changes sign while touching the output section (trap 2)");
    }
    runEnd_ = t1;
    first_ = false;
  }

  // 반환 시간 창과 겹치는 묶음이 정확히 하나이고, 그 앞에서 끝난 지정 방향 통과가 returns-1 개여야 한다
  void finish() {
    if (runSign_ != 0) closeRun(0);
    if (windowRuns_ != 1)
      throw std::runtime_error("crossing: " + std::to_string(windowRuns_) +
                               " touching runs overlap the return-time window (trap 2)");
    if (countedBefore_ != returns_ - 1)
      throw std::runtime_error("crossing: " + std::to_string(countedBefore_) +
                               " counted crossings precede the returned one, expected " +
                               std::to_string(returns_ - 1) + " (trap 2)");
  }

  // finish() 를 통과한 뒤의 세기 결과 — 덮개 기록이 첫 통과 구조를 저장값으로 다시 보게 내보낸다
  int windowRuns() const { return windowRuns_; }
  int countedBefore() const { return countedBefore_; }
  int runsBefore() const { return runsBefore_; }
  const Interval& windowRun() const { return windowRun_; }

 private:
  // after: 묶음 뒤 단계의 s 부호(0 = 흐름 관이 묶음 안에서 끝남)
  void closeRun(int after) {
    const bool inWindow = !(runStart_.leftBound() > window_.rightBound()) &&
                          !(runEnd_.rightBound() < window_.leftBound());
    if (inWindow) {
      if (runSign_ != dir_) throw std::runtime_error("crossing: the run at the return time has the wrong direction (trap 2)");
      if (runFromStart_ && onSection_)
        throw std::runtime_error("crossing: return-time window overlaps the departure from the section (trap 2)");
      if (++windowRuns_ == 1) countedBefore_ = counted_;
      if (windowRuns_ == 1) windowRun_ = A::hull(runStart_, runEnd_);
    } else if (windowRuns_ == 0) {
      ++runsBefore_;
      if (runFromStart_ && onSection_) {
        if (after != runSign_) throw std::runtime_error("crossing: departure from the section is inconsistent (trap 2)");
      } else if (before_ == 0) {
        if (runSign_ == dir_)
          throw std::runtime_error("crossing: initial set straddles the output section moving in the counted direction (trap 2)");
      } else if (before_ != after) {
        if (after != runSign_) throw std::runtime_error("crossing: sign change disagrees with ds/dt (trap 2)");
        if (runSign_ == dir_) ++counted_;
      }
    }
    runSign_ = 0;
  }

  const int dir_, returns_;
  const Interval window_;
  int before_;
  const bool onSection_;
  bool first_ = true, runFromStart_ = false;
  int runSign_ = 0, counted_ = 0, countedBefore_ = -1, windowRuns_ = 0, runsBefore_ = 0;
  Interval runStart_, runEnd_, windowRun_;
};

// ---------------------------------------------------------------- 계산 본체

template <class A>
class Engine {
 public:
  using Interval = typename A::Interval;
  using IVector = typename A::IVector;
  using IMatrix = typename A::IMatrix;
  using IMap = typename A::IMap;

  explicit Engine(const Request<A>& req)
      : req_(req), sys_(req.m1, req.m2, req.m3, req.J, req.h), f6_(sys_.vectorField(kPhaseDim)) {}

  std::string run(const IVector& box) {
    Writer<A> w;
    if (req_.mode == "vf") vfMode(box, w);
    else if (req_.mode == "chart") chartMode(box, w);
    else if (req_.mode == "c0") c0Mode(box, w);
    else if (req_.mode == "c1") c1Mode(box, w);
    else throw std::runtime_error("request: unknown mode " + req_.mode);
    return w.out.str();
  }

 private:
  struct InitialSet {
    IVector center, r0, rest;  // y in center + C*r0 + rest
    IMatrix C;
    IMatrix dPsi;     // d y / d w (dim x 4)
    Interval disc;    // 차트 판별식 D 의 감싸기
    IVector yHull;    // Psi(W) 의 감싸기
  };

  void configure(typename A::Solver& solver) const {
    if (req_.fixedStep) solver.setStep(req_.step);
  }

  // 입력 틀 좌표 w 상자 -> 위상공간 이중집합 (평균값 형식, 함정 ⑥ 이 아니라 상자 전체를 감싼다)
  InitialSet initialSet(const IVector& W, int dim) const {
    const SectionSpec& s = req_.in;
    IVector wMid(4), wDev(4);
    for (int i = 0; i < 4; ++i) { wMid[i] = A::thin(W[i]); wDev[i] = W[i] - wMid[i]; }
    const IVector U = req_.frameInCenter + req_.frameInMatrix * W;
    const IVector uMid = req_.frameInCenter + req_.frameInMatrix * wMid;
    InitialSet set;
    Interval discMid;
    const IVector yMid6 = sys_.chart(uMid, s, req_.cIn, discMid);
    const IVector yBox6 = sys_.chart(U, s, req_.cIn, set.disc);
    const IMatrix dPhi = sys_.chartDerivative(f6_, yBox6, s, set.disc);
    const IMatrix dPsi6 = dPhi * req_.frameInMatrix;

    set.dPsi = IMatrix(dim, 4);
    set.C = IMatrix(dim, dim);
    set.r0 = IVector(dim);
    set.center = IVector(dim);
    set.rest = IVector(dim);
    set.yHull = IVector(dim);
    IVector x0(dim);
    for (int i = 0; i < kPhaseDim; ++i) {
      Interval lin = A::fromInteger(0);
      for (int j = 0; j < 4; ++j) {
        set.dPsi[i][j] = dPsi6[i][j];
        const Interval cmid = A::thin(dPsi6[i][j]);
        set.C[i][j] = cmid;
        lin += (dPsi6[i][j] - cmid) * wDev[j];
      }
      x0[i] = yMid6[i] + lin;
      set.yHull[i] = yBox6[i];
    }
    // 남는 열(단면 방향, 운동량 방향, 회전각)은 반경 0 의 자리채움 — C 를 정칙으로 둔다
    std::vector<int> fillers = {s.k, s.m};
    if (dim == 7) fillers.push_back(kAngleIndex);
    for (size_t j = 0; j < fillers.size(); ++j) set.C[fillers[j]][4 + static_cast<int>(j)] = A::fromInteger(1);
    for (int j = 0; j < 4; ++j) set.r0[j] = wDev[j];
    for (int i = 0; i < dim; ++i) { set.center[i] = A::thin(x0[i]); set.rest[i] = x0[i] - set.center[i]; }
    return set;
  }

  typename A::Section section(int dim) const {
    return typename A::Section(dim, req_.out.k, req_.cOut);
  }

  static capd::poincare::CrossingDirection crossing(int dir) {
    return dir > 0 ? capd::poincare::MinusPlus : capd::poincare::PlusMinus;
  }

  // 출력 틀: w' = Minv * (Pi_out y - c_out). 행 4.. 는 단면·운동량·회전각 자리
  void outputFrame(int dim, IMatrix& Aff, IVector& c) const {
    const SectionSpec& s = req_.out;
    Aff = IMatrix(dim, dim);
    c = IVector(dim);
    for (int i = 0; i < 4; ++i)
      for (int l = 0; l < 4; ++l) Aff[i][s.freeIdx[l]] = req_.frameOutInverse[i][l];
    Aff[4][s.k] = A::fromInteger(1);
    Aff[5][s.m] = A::fromInteger(1);
    if (dim == 7) Aff[6][kAngleIndex] = A::fromInteger(1);
    for (int l = 0; l < 4; ++l) c[s.freeIdx[l]] = req_.frameOutCenter[l];
  }

  IVector frameImageFromHull(const IVector& yHull) const {
    const SectionSpec& s = req_.out;
    IVector d(4);
    for (int l = 0; l < 4; ++l) d[l] = yHull[s.freeIdx[l]] - req_.frameOutCenter[l];
    return req_.frameOutInverse * d;
  }

  // 흐름 관(함정 ④): [0, tEnd] 를 덮을 때까지 C0 로 다시 적분하며 단계 감싸기마다 r_ij 상·하한을 모으고,
  // 같은 단계로 출력 단면 통과를 센다(함정 ②, CrossingCounter)
  void tube(const InitialSet& init, int dim, const Interval& returnTime, Writer<A>& w) const {
    IMap f = sys_.vectorField(dim);
    typename A::Solver solver(f, req_.order);
    configure(solver);
    typename A::C0Set set(init.center, init.C, init.r0, init.rest);
    Interval r12 = init.yHull[0], r13, r23;
    {
      const auto d = sys_.distances(init.yHull);
      r13 = d.r13; r23 = d.r23;
    }
    const int k = req_.out.k;
    const Interval s0 = init.yHull[k] - req_.cOut;
    // 같은 좌표·같은 값 토큰이면 초기 집합은 차트가 y_k=c 로 만든 단면 위의 집합이다
    const bool onSection = req_.in.k == k && req_.cIn.leftBound() == req_.cOut.leftBound() &&
                           req_.cIn.rightBound() == req_.cOut.rightBound();
    CrossingCounter<A> crossings(req_.out.dir, req_.returns, returnTime,
                                 s0.leftBound() > 0 ? 1 : (s0.rightBound() < 0 ? -1 : 0), onSection);
    long steps = 0;
    // PoincareMap·TimeMap 과 같이 첫 보폭을 집합에 맞춰 정한다 — 생략하면 기본 보폭으로 감싸기가 터진다
    if (!req_.fixedStep) solver.getStepControl().init(solver, set.getCurrentTime(), set);
    while (set.getCurrentTime().leftBound() < returnTime.rightBound()) {
      const Interval t0 = set.getCurrentTime();
      try {
        solver(set);
      } catch (std::exception& e) {
        std::ostringstream where;
        where << "step " << steps << " at t=" << A::fmtLo(set.getCurrentTime()) << " of T=" << A::fmtHi(returnTime)
              << ": " << firstLine(e.what());
        throw std::runtime_error(where.str());
      }
      const IVector enclosure = set.getLastEnclosure();
      const auto d = sys_.distances(enclosure);
      r12 = intervalHull(r12, d.r12);
      r13 = intervalHull(r13, d.r13);
      r23 = intervalHull(r23, d.r23);
      crossings.step(enclosure[k] - req_.cOut, f(enclosure)[k], t0, set.getCurrentTime());
      if (++steps > 2000000) throw std::runtime_error("tube: too many steps");
    }
    crossings.finish();
    w.scalar("tube_r12", r12);
    w.scalar("tube_r13", r13);
    w.scalar("tube_r23", r23);
    w.out << "tube_steps " << steps << '\n';
    w.scalar("tube_tend", set.getCurrentTime());
    w.out << "crossings " << crossings.windowRuns() << ' ' << crossings.countedBefore() << ' ' << crossings.runsBefore()
          << '\n';
    w.scalar("crossing_run", crossings.windowRun());
  }

  // 도착 차트의 근 선택(branch_out) 확인용: dH/dy_m 를 상 위에서 감싼다
  void branchOut(const IVector& yHull, Writer<A>& w) const {
    const IVector g = sys_.gradientH(f6_, yHull);
    w.scalar("branch_out", A::fromInteger(req_.out.branch) * g[req_.out.m]);
  }

  void vfMode(const IVector& y, Writer<A>& w) const {
    w.vector("vf", f6_(y), kPhaseDim);
    w.scalar("energy", sys_.hamiltonian(y));
    const auto d = sys_.distances(y);
    w.scalar("r12", d.r12);
    w.scalar("r13", d.r13);
    w.scalar("r23", d.r23);
  }

  void chartMode(const IVector& W, Writer<A>& w) const {
    const InitialSet init = initialSet(W, kPhaseDim);
    w.vector("chart_hull", init.yHull, kPhaseDim);
    w.matrix("chart_dpsi", init.dPsi, kPhaseDim, 4);
    w.scalar("disc", init.disc);
    w.scalar("energy", sys_.hamiltonian(init.yHull));
  }

  void c0Mode(const IVector& W, Writer<A>& w) const {
    const int dim = req_.rotation ? 7 : kPhaseDim;
    const InitialSet init = stage("chart", [&] { return initialSet(W, dim); });
    w.scalar("disc", init.disc);
    IMap f = sys_.vectorField(dim);
    auto sec = section(dim);

    // (1) 감싸기 상자와 반환 시간
    typename A::Solver solver(f, req_.order);
    configure(solver);
    typename A::PoincareMap pm(solver, sec, crossing(req_.out.dir));
    typename A::C0Set setHull(init.center, init.C, init.r0, init.rest);
    Interval returnTime;
    const IVector yHull = stage("c0 hull poincare", [&] { return pm(setHull, returnTime, req_.returns); });
    w.vector("hull", yHull, dim);
    w.scalar("return_time", returnTime);

    // (2) 출력 틀 좌표에서 집합 표현을 살린 감싸기(래핑 억제)
    typename A::Solver solver2(f, req_.order);
    configure(solver2);
    typename A::PoincareMap pm2(solver2, sec, crossing(req_.out.dir));
    typename A::C0Set setAff(init.center, init.C, init.r0, init.rest);
    IMatrix Aff;
    IVector c;
    outputFrame(dim, Aff, c);
    Interval returnTime2;
    const IVector affine = stage("c0 affine poincare", [&] { return pm2(setAff, c, Aff, returnTime2, req_.returns); });
    // 교집합은 두 상이 같은 통과의 상일 때만 그 통과를 감싼다 — 같은 설정·같은 집합의 결정론적 계산이라 반환 시각이 비트 단위로 같아야 한다
    if (!(returnTime2.leftBound() == returnTime.leftBound() && returnTime2.rightBound() == returnTime.rightBound()))
      throw std::runtime_error("c0: affine and hull return times differ — report as engine defect");
    IVector image = frameImageFromHull(yHull);
    for (int i = 0; i < 4; ++i) {
      Interval both;
      if (!intersection(image[i], affine[i], both))
        throw std::runtime_error("c0: affine and hull enclosures are disjoint — report as engine defect");
      image[i] = both;
    }
    w.vector("image", image, 4);
    if (dim == 7) w.scalar("theta", affine[kAngleIndex]);
    stage("branch check", [&] { branchOut(yHull, w); });
    stage("tube", [&] { tube(init, dim, returnTime, w); });
  }

  void c1Mode(const IVector& W, Writer<A>& w) const {
    const InitialSet init = stage("chart", [&] { return initialSet(W, kPhaseDim); });
    w.scalar("disc", init.disc);
    IMap f = f6_;
    auto sec = section(kPhaseDim);
    typename A::Solver solver(f, req_.order);
    configure(solver);
    typename A::PoincareMap pm(solver, sec, crossing(req_.out.dir));
    typename A::C1Set set(init.center, init.C, init.r0, init.rest);
    IMatrix dF(kPhaseDim, kPhaseDim);
    Interval returnTime;
    const IVector yHull = stage("c1 poincare", [&] { return pm(set, dF, returnTime, req_.returns); });
    const IMatrix dP6 = stage("c1 computeDP", [&] { return pm.computeDP(yHull, dF, returnTime); });
    IMatrix proj(4, kPhaseDim);
    for (int l = 0; l < 4; ++l) proj[l][req_.out.freeIdx[l]] = A::fromInteger(1);
    const IMatrix dPw = req_.frameOutInverse * (proj * (dP6 * init.dPsi));
    w.vector("hull", yHull, kPhaseDim);
    w.scalar("return_time", returnTime);
    w.vector("image", frameImageFromHull(yHull), 4);
    w.matrix("deriv", dPw, 4, 4);
    w.matrix("dp6", dP6, kPhaseDim, kPhaseDim);
    stage("branch check", [&] { branchOut(yHull, w); });
    stage("tube", [&] { tube(init, kPhaseDim, returnTime, w); });
  }

  const Request<A>& req_;
  ReducedSystem<A> sys_;
  mutable IMap f6_;
};

std::string firstLine(const std::string& s) {
  const auto nl = s.find('\n');
  std::string line = s.substr(0, nl);
  for (char& ch : line) if (ch == '\t') ch = ' ';
  return line;
}

template <class A>
int serve(Tokens& t, int precisionBits) {
  A::configure(precisionBits);
  const Request<A> req = parseRequest<A>(t);
  Engine<A> engine(req);
  std::cout << "T3CAP 1 arith " << A::name() << " boxes " << req.boxes.size() << '\n';
  for (size_t i = 0; i < req.boxes.size(); ++i) {
    const auto start = std::chrono::steady_clock::now();
    std::string body, status = "ok";
    try {
      body = engine.run(req.boxes[i]);
    } catch (std::exception& e) {
      status = "fail " + firstLine(e.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "box " << i << ' ' << status << '\n' << body << "seconds " << secs << "\nend\n";
  }
  std::cout << "done\n";
  return 0;
}

}  // namespace t3cap

int main() {
  std::ios::sync_with_stdio(false);
  t3cap::Tokens t;
  std::string tok;
  while (std::cin >> tok) t.items.push_back(tok);
  try {
    if (t.next() != "T3CAP") throw std::runtime_error("request: missing T3CAP header");
    if (t.next() != "arith") throw std::runtime_error("request: second key must be arith");
    const std::string arith = t.next();
    if (arith == "double") return t3cap::serve<t3cap::DoubleArith>(t, 53);
#ifdef __HAVE_MPFR__
    if (arith.rfind("mp:", 0) == 0) return t3cap::serve<t3cap::MpArith>(t, std::stoi(arith.substr(3)));
#endif
    throw std::runtime_error("request: unsupported arith " + arith);
  } catch (std::exception& e) {
    std::cout << "error " << t3cap::firstLine(e.what()) << '\n';
    return 2;
  }
}
