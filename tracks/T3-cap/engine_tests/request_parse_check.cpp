// 요청 해석 점검: 표준입력의 요청들(각각 "T3CAP" 으로 시작)을 엔진과 같은 경로(산술 설정 → parseRequest)로 읽기만 하고
// 적분하지 않는다. 요청마다 "ok <산술> <상자 수>" 또는 "fail <산술> <사유>" 를 한 줄씩 낸다. 하나라도 실패하면 종료 코드 1.
// 쓰임: T6·T7 이 만드는 요청 토큰이 엔진의 끝점 가드에서 거부되지 않는지 본 인증 전에 확인한다.
#define main t3cap_engine_main_unused
#include "../engine/t3cap_engine.cpp"
#undef main

#include <iostream>

namespace {

template <class A>
std::string parsed(t3cap::Tokens& t, int precisionBits) {
  A::configure(precisionBits);
  const t3cap::Request<A> req = t3cap::parseRequest<A>(t);
  return "ok " + A::name() + " " + std::to_string(req.boxes.size());
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::vector<std::string> all;
  for (std::string tok; std::cin >> tok;) all.push_back(tok);
  int failed = 0;
  for (size_t i = 0; i < all.size();) {
    size_t j = i + 1;
    while (j < all.size() && all[j] != "T3CAP") ++j;
    t3cap::Tokens t;
    t.items.assign(all.begin() + static_cast<long>(i), all.begin() + static_cast<long>(j));
    std::string arith = "?";
    try {
      if (t.next() != "T3CAP" || t.next() != "arith") throw std::runtime_error("request: bad header");
      arith = t.next();
      if (arith == "double") {
        std::cout << parsed<t3cap::DoubleArith>(t, 53) << '\n';
#ifdef __HAVE_MPFR__
      } else if (arith.rfind("mp:", 0) == 0) {
        std::cout << parsed<t3cap::MpArith>(t, std::stoi(arith.substr(3))) << '\n';
#endif
      } else {
        throw std::runtime_error("request: unsupported arith " + arith);
      }
    } catch (std::exception& e) {
      ++failed;
      std::cout << "fail " << arith << ' ' << t3cap::firstLine(e.what()) << '\n';
    }
    i = j;
  }
  return failed ? 1 : 0;
}
