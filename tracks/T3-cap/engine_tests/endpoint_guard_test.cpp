// 요청 끝점 가드 시험: 16진 토큰은 그 실행의 산술로 정확히 나타날 때만 받고(그 값 그대로), 반올림이 필요한 토큰·잘못된
// 꼴은 거부해야 한다. MP 산술은 53비트 미만을 거부해야 한다. 기대값은 IEEE double 의 정의(홀수 가수 53비트 이하, 최하위
// 비트 지수 −1074 이상, 최상위 비트 지수 1023 이하)와 MPFR 수의 정의(홀수 가수가 설정 비트 이하)에서 정했다.
// 엔진 소스는 그대로 포함한다. 실패가 있으면 종료 코드 1.
#define main t3cap_engine_main_unused
#include "../engine/t3cap_engine.cpp"
#undef main

#include <cstdio>
#include <limits>

namespace {

struct Accept {
  const char* token;
  double value;
};

// 받아들이면 빈 문자열, 거부하면 그 사유
std::string parseVerdict(const std::string& token, double& value) {
  try {
    value = t3cap::exactHexDouble(token);
    return "";
  } catch (std::exception& e) {
    return e.what();
  }
}

#ifdef __HAVE_MPFR__
// 설정 비트로 MP 16진 끝점을 읽는다. 받아들이면 빈 문자열과 그 값, 거부하면 그 사유
std::string mpHexVerdict(int bits, const std::string& token, capd::MpFloat& value) {
  try {
    t3cap::MpArith::configure(bits);
    const auto point = t3cap::parseEndpoint<t3cap::MpArith>(token);
    if (!(point.leftBound() == point.rightBound())) return "not a point interval";
    value = point.leftBound();
    return "";
  } catch (std::exception& e) {
    return e.what();
  }
}

capd::MpFloat powerOfTwo(int n) {
  capd::MpFloat v(1.0);
  for (int k = 0; k < (n < 0 ? -n : n); ++k) v = n < 0 ? v / capd::MpFloat(2.0) : v * capd::MpFloat(2.0);
  return v;
}

std::string mpVerdict(int bits) {
  try {
    t3cap::MpArith::configure(bits);
    return "";
  } catch (std::exception& e) {
    return e.what();
  }
}
#endif

}  // namespace

int main() {
  const double tiny = std::numeric_limits<double>::denorm_min();
  const std::vector<Accept> accepted = {
      {"0x1.8p+1", 3.0},
      {"0x1.8000p+1", 3.0},
      {"0x0001.8p+1", 3.0},
      {"0X1.8P+1", 3.0},
      {"+0x1.8p1", 3.0},
      {"0x10", 16.0},
      {"0x1p+0", 1.0},
      {"0x.8", 0.5},
      {"0x.8p1", 1.0},
      {"0x1.000000000000000000000000000000000000000000000000000000000p+0", 1.0},  // 64자
      {"-0x1.999999999999ap-4", -0.1},
      {"0x1.0000000000001p+0", 1.0 + std::numeric_limits<double>::epsilon()},
      {"0x1p-1074", tiny},
      {"0x0.0000000000001p-1022", tiny},
      {"0x1.fffffffffffffp+1023", std::numeric_limits<double>::max()},
      {"-0x0p+0", -0.0},
  };
  const std::vector<const char*> notExact = {
      "0x1.00000000000001p+0",     // 57 비트 가수
      "0x1.00000000000008p+0",     // 1+2^-53, 54 비트
      "0x1p+1024",                 // 최댓값 위
      "0x1p-1075",                 // 가장 작은 비정규 수 아래
      "0x1.8p-1074",               // 최하위 비트가 2^-1075
      "0x20000000000001",          // 2^53+1
  };
  const std::vector<const char*> malformed = {"0x", "0x.", "0xg1", "0x1p", "0x1p+", "0x1.2.3", "1.5", "0x1p+123456789",
                                              "0x1p1x", "-", "",
                                              "0x1.0000000000000000000000000000000000000000000000000000000000p+0"};  // 65자
  int failed = 0, total = 0;
  const auto report = [&](bool ok, const std::string& what, const std::string& detail) {
    ++total;
    failed += !ok;
    std::printf("%s %s%s%s\n", ok ? "ok" : "FAIL", what.c_str(), detail.empty() ? "" : " | ", detail.c_str());
  };
  for (const Accept& a : accepted) {
    double v = 0;
    const std::string reason = parseVerdict(a.token, v);
    // 값과 0 의 부호까지 같아야 한다
    const bool same = reason.empty() && v == a.value && std::signbit(v) == std::signbit(a.value);
    report(same, std::string("accept exact double ") + a.token, reason);
  }
  for (const char* t : notExact) {
    double v = 0;
    const std::string reason = parseVerdict(t, v);
    report(reason.find("not exactly a double") != std::string::npos, std::string("reject rounding token ") + t, reason);
  }
  for (const char* t : malformed) {
    double v = 0;
    const std::string reason = parseVerdict(t, v);
    report(reason.find("bad hex float") != std::string::npos, std::string("reject malformed token '") + t + "'", reason);
  }
#ifdef __HAVE_MPFR__
  report(mpVerdict(52).find("at least 53 bits") != std::string::npos, "mp precision 52 bits is refused", mpVerdict(52));
  report(mpVerdict(53).empty(), "mp precision 53 bits is accepted", mpVerdict(53));
  // 53 비트에서 16진 끝점과 2^53 이하 p/q 의 정수 부분이 점 구간으로 정확히 들어온다
  const auto hexPoint = t3cap::parseEndpoint<t3cap::MpArith>("0x1.0000000000001p+0");
  const auto bigInteger = t3cap::parseEndpoint<t3cap::MpArith>("9007199254740992/1");
  report(hexPoint.leftBound() == hexPoint.rightBound() && hexPoint.leftBound() == capd::MpFloat(1.0 + std::numeric_limits<double>::epsilon()),
         "mp:53 hex endpoint is an exact point", "");
  report(bigInteger.leftBound() == bigInteger.rightBound() && bigInteger.leftBound() == capd::MpFloat(9007199254740992.0),
         "mp:53 p/q endpoint with p=2^53 is an exact point", "");
  // MP 실행은 홀수 가수가 설정 비트 이하인 16진 수를 정확히 받는다(double 범위·53비트에 묶이지 않는다)
  const std::string bits57 = "0x1.00000000000001p+0", bits113 = "0x1.0000000000000000000000000001p+0",
                    bits129 = "0x1.00000000000000000000000000000001p+0";
  capd::MpFloat v;
  std::string why = mpHexVerdict(53, bits57, v);
  report(why.find("not exactly an mp:53 number") != std::string::npos, "mp:53 refuses a 57-bit hex endpoint", why);
  why = mpHexVerdict(56, bits57, v);
  report(why.find("not exactly an mp:56 number") != std::string::npos, "mp:56 refuses a 57-bit hex endpoint", why);
  why = mpHexVerdict(57, bits57, v);
  report(why.empty() && v - capd::MpFloat(1.0) == powerOfTwo(-56), "mp:57 reads a 57-bit hex endpoint exactly", why);
  why = mpHexVerdict(128, bits113, v);
  report(why.empty() && v - capd::MpFloat(1.0) == powerOfTwo(-112), "mp:128 reads a 113-bit hex endpoint exactly", why);
  why = mpHexVerdict(128, bits129, v);
  report(why.find("not exactly an mp:128 number") != std::string::npos, "mp:128 refuses a 129-bit hex endpoint", why);
  why = mpHexVerdict(128, "0x1p+5000", v);
  report(why.empty() && v == powerOfTwo(5000), "mp:128 reads 2^5000 exactly (outside the double range)", why);
  why = mpHexVerdict(128, "-0x0p+0", v);
  report(why.empty() && v == capd::MpFloat(0.0), "mp:128 reads a signed zero", why);
  why = mpHexVerdict(128, "0x1p1x", v);
  report(why.find("bad hex float") != std::string::npos, "mp:128 refuses a malformed token", why);
#endif
  std::printf("%d/%d passed\n", total - failed, total);
  return failed ? 1 : 0;
}
