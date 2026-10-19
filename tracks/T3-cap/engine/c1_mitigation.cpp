// C1 과대평가 완화책 실측용: 엔진과 같은 요청을 받되 C1 집합 표현만 바꿔 돌린다. 인자: rect2 | pped2 | horect2
// 엔진 소스를 그대로 포함해 계산 경로를 복제하지 않는다(main 만 이름을 바꿔 가린다).
#define main t3cap_engine_main_unused
#include "t3cap_engine.cpp"
#undef main

namespace {

struct Pped2Arith : t3cap::DoubleArith {
  using C1Set = capd::C1Pped2Set;
};

struct HORect2Arith : t3cap::DoubleArith {
  using C1Set = capd::C1HORect2Set;
};

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  const std::string policy = argc > 1 ? argv[1] : "rect2";
  t3cap::Tokens t;
  std::string tok;
  while (std::cin >> tok) t.items.push_back(tok);
  try {
    if (t.next() != "T3CAP" || t.next() != "arith" || t.next() != "double")
      throw std::runtime_error("c1_mitigation: expects a double-arith T3CAP request");
    if (policy == "rect2") return t3cap::serve<t3cap::DoubleArith>(t, 53);
    if (policy == "pped2") return t3cap::serve<Pped2Arith>(t, 53);
    if (policy == "horect2") return t3cap::serve<HORect2Arith>(t, 53);
    throw std::runtime_error("c1_mitigation: unknown policy " + policy);
  } catch (std::exception& e) {
    std::cout << "error " << t3cap::firstLine(e.what()) << '\n';
    return 2;
  }
}
