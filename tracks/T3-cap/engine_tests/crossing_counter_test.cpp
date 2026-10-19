// 출력 단면 통과 세기(CrossingCounter, 함정 ②) 시험. 단계별 감싸기의 단면 함수 값·ds/dt·시간을 직접 넣어
// 「t>0 의 returns 번째 지정 방향 통과」 규칙과 받아들인 뒤 내보내는 세기 결과를 본다. 통합 사례는 각본 단계열을 내는
// 가짜 적분기로 Engine::run 을 돌려 tube() 의 finish() 호출과 출력 줄, c0 의 두 반환 창 일치 단언까지 확인한다.
// 엔진 소스는 그대로 포함한다(main 만 이름을 바꿔 가린다). 실패가 있으면 종료 코드 1.
#define main t3cap_engine_main_unused
#include "../engine/t3cap_engine.cpp"
#undef main

#include <cstdio>

namespace {

using I = capd::interval;
using capd::IMatrix;
using capd::IVector;

// ---------------------------------------------------------------- CrossingCounter 직접 시험

struct Step {
  I s, dsdt, t0, t1;
};

Step above(double t0, double t1) { return {I(0.1, 0.2), I(1.0), I(t0), I(t1)}; }
Step below(double t0, double t1) { return {I(-0.2, -0.1), I(1.0), I(t0), I(t1)}; }
Step touch(double glo, double ghi, double t0, double t1) { return {I(-1e-3, 1e-3), I(glo, ghi), I(t0), I(t1)}; }
// 단면 위(s=0)에서 위로 떠나는 첫 단계
Step depart(double t0, double t1) { return {I(0.0, 0.04), I(0.3, 0.5), I(t0), I(t1)}; }

// 받아들이면 빈 문자열, 거부하면 그 사유
std::string counterVerdict(int dir, int returns, I window, int initialSign, bool onSection, const std::vector<Step>& steps) {
  try {
    t3cap::CrossingCounter<t3cap::DoubleArith> counter(dir, returns, window, initialSign, onSection);
    for (const Step& st : steps) counter.step(st.s, st.dsdt, st.t0, st.t1);
    counter.finish();
    return "";
  } catch (std::exception& e) {
    return e.what();
  }
}

// 받아들인 뒤 내보내는 세기 결과(창 묶음 수, 앞선 지정 방향 통과 수, 창 앞 닫힌 묶음 수, 창 묶음 시각)가 기대와 같으면 빈 문자열
std::string counterReport(int dir, int returns, I window, int initialSign, bool onSection, const std::vector<Step>& steps,
                          int windowRuns, int countedBefore, int runsBefore, I windowRun) {
  try {
    t3cap::CrossingCounter<t3cap::DoubleArith> counter(dir, returns, window, initialSign, onSection);
    for (const Step& st : steps) counter.step(st.s, st.dsdt, st.t0, st.t1);
    counter.finish();
    const I run = counter.windowRun();
    if (counter.windowRuns() == windowRuns && counter.countedBefore() == countedBefore &&
        counter.runsBefore() == runsBefore && run.leftBound() == windowRun.leftBound() &&
        run.rightBound() == windowRun.rightBound())
      return "";
    char buf[160];
    std::snprintf(buf, sizeof buf, "reported %d %d %d [%g, %g]", counter.windowRuns(), counter.countedBefore(),
                  counter.runsBefore(), run.leftBound(), run.rightBound());
    return buf;
  } catch (std::exception& e) {
    return e.what();
  }
}

// ---------------------------------------------------------------- 각본 적분기로 Engine::run 을 거치는 통합 시험

// 단계 감싸기는 축소 좌표 (r, x, y, pr, px, py) = (5, 0, y, 0, 0.4, py). x=0 이면 ẏ = py/μ₂ 라 py 부호가 ds/dt 부호다
struct ScriptStep {
  I y, py;
  double t1;
};

std::vector<ScriptStep> g_steps;
size_t g_next = 0;
I g_window;
I g_affineWindow;  // 아핀 상 호출이 돌려주는 반환 창(보통 g_window 와 같다)
std::string g_output;

IVector hullPoint() {
  IVector v(6);
  v[0] = I(5.0);
  v[4] = I(0.4);
  v[5] = I(0.4);
  return v;
}

struct ScriptedSet {
  ScriptedSet(const IVector&, const IMatrix&, const IVector&, const IVector&) {}
  const I& getCurrentTime() const { return time; }
  const IVector& getLastEnclosure() const { return enclosure; }
  I time = I(0.0);
  IVector enclosure = IVector(6);
};

struct ScriptedSolver {
  struct StepControl {
    template <class... T>
    void init(T&&...) {}
  };
  template <class Map>
  ScriptedSolver(Map&, int) {}
  void setStep(const I&) {}
  StepControl& getStepControl() { return control; }
  void operator()(ScriptedSet& set) {
    if (g_next >= g_steps.size()) throw std::runtime_error("script exhausted");
    const ScriptStep& st = g_steps[g_next++];
    IVector e = hullPoint();
    e[2] = st.y;
    e[5] = st.py;
    set.enclosure = e;
    set.time = I(st.t1);
  }
  StepControl control;
};

// CAPD 대신 각본의 반환 창과 상을 돌려준다(통과 세기는 흐름 관 단계만 본다)
struct ScriptedPoincareMap {
  template <class Solver, class Section, class Direction>
  ScriptedPoincareMap(Solver&, Section&, Direction) {}
  IVector operator()(ScriptedSet&, I& returnTime, int) {
    returnTime = g_window;
    return hullPoint();
  }
  IVector operator()(ScriptedSet&, const IVector&, const IMatrix&, I& returnTime, int) {
    returnTime = g_affineWindow;
    IVector wide(6);
    for (int i = 0; i < 6; ++i) wide[i] = I(-1e6, 1e6);
    return wide;
  }
  IVector operator()(ScriptedSet&, IMatrix& dF, I& returnTime, int) {
    returnTime = g_window;
    dF = IMatrix::Identity(6);
    return hullPoint();
  }
  IMatrix computeDP(const IVector&, const IMatrix&, const I&) { return IMatrix::Identity(6); }
};

struct ScriptedArith : t3cap::DoubleArith {
  using Solver = ScriptedSolver;
  using PoincareMap = ScriptedPoincareMap;
  using C0Set = ScriptedSet;
  using C1Set = ScriptedSet;
};

// 8자 궤도 Σ+ 통과점 둘레 반경 1e-6 상자, 입력·출력 모두 syzygy 단면(같은 좌표·같은 값이라 출발 집합은 단면 위)
std::string request(const std::string& mode, int outDir, int returns) {
  const std::string center =
      "5148567982527431/1000000000000000 5148567982527431/1000000000000000 0/1 0/1 0/1 0/1 "
      "432690650760826/1000000000000000 432690650760826/1000000000000000 ";
  std::string identity;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) identity += i == j ? "1/1 1/1 " : "0/1 0/1 ";
  const std::string d = std::to_string(outDir);
  return "mode " + mode + " order 20 step adaptive masses 1/1 1/1 1/1 1/1 1/1 1/1 energy -1/2 -1/2 angmom 0/1 0/1 " +
         "section_in 2 1 5 1 0/1 0/1 section_out 2 " + d + " 5 " + d + " 0/1 0/1 returns " + std::to_string(returns) +
         " rotation 0 frame_in " + center + identity + "frame_out " + center + identity +
         "boxes 1 4 -1/1000000 1/1000000 -1/1000000 1/1000000 -1/1000000 1/1000000 -1/1000000 1/1000000 end";
}

std::string engineVerdict(const std::string& mode, int outDir, int returns, I window, const std::vector<ScriptStep>& steps,
                          I affineWindow) {
  g_steps = steps;
  g_next = 0;
  g_window = window;
  g_affineWindow = affineWindow;
  g_output.clear();
  t3cap::Tokens tokens;
  std::istringstream in(request(mode, outDir, returns));
  for (std::string tok; in >> tok;) tokens.items.push_back(tok);
  try {
    const t3cap::Request<ScriptedArith> req = t3cap::parseRequest<ScriptedArith>(tokens);
    t3cap::Engine<ScriptedArith> engine(req);
    g_output = engine.run(req.boxes[0]);
    return "";
  } catch (std::exception& e) {
    return e.what();
  }
}

std::string engineVerdict(const std::string& mode, int outDir, int returns, I window, const std::vector<ScriptStep>& steps) {
  return engineVerdict(mode, outDir, returns, window, steps, window);
}

// 엔진이 받아들인 뒤 출력의 tube_tend 다음에 세기 결과 두 줄이 기대대로 실렸으면 빈 문자열
std::string engineReport(const std::string& mode, int outDir, int returns, I window, const std::vector<ScriptStep>& steps,
                         int windowRuns, int countedBefore, int runsBefore, double runLo, double runHi) {
  const std::string reason = engineVerdict(mode, outDir, returns, window, steps);
  if (!reason.empty()) return reason;
  char lines[200];
  std::snprintf(lines, sizeof lines, "\ncrossings %d %d %d\ncrossing_run %a %a\n", windowRuns, countedBefore, runsBefore,
                runLo, runHi);
  const size_t tend = g_output.find("tube_tend ");
  const size_t next = tend == std::string::npos ? tend : g_output.find('\n', tend);
  return next != std::string::npos && g_output.compare(next, std::string(lines).size(), lines) == 0
             ? ""
             : "expected '" + std::string(lines) + "' after tube_tend in output";
}

ScriptStep leave(double t1) { return {I(0.0, 0.04), I(0.4), t1}; }
ScriptStep aboveS(double t1) { return {I(0.1, 0.2), I(0.4), t1}; }
ScriptStep belowS(double t1) { return {I(-0.2, -0.1), I(0.4), t1}; }
ScriptStep crossDown(double t1) { return {I(-1e-3, 1e-3), I(-0.4), t1}; }
ScriptStep crossUp(double t1) { return {I(-1e-3, 1e-3), I(0.4), t1}; }
ScriptStep crossFlat(double t1) { return {I(-1e-3, 1e-3), I(-0.1, 0.2), t1}; }

}  // namespace

int main() {
  // 같은 단면 반환(P̃ 꼴): 위로 떠남 → 아래로(안 셈) → 위로(셈 1) → 아래로 → 위로(셈 2)
  const std::vector<Step> twoUp = {depart(0, 0.1),        above(0.1, 4.2),  touch(-0.5, -0.3, 4.2, 4.5),
                                   below(4.5, 8.5),       touch(0.3, 0.5, 8.5, 8.9), above(8.9, 13.0),
                                   touch(-0.5, -0.3, 13.0, 13.3), below(13.3, 17.2), touch(0.3, 0.5, 17.2, 17.6)};
  const std::vector<ScriptStep> twoUpScript = {leave(0.1),     aboveS(4.2), crossDown(4.5), belowS(8.5),  crossUp(8.9),
                                               aboveS(13.0),   crossDown(13.3), belowS(17.2), crossUp(17.6)};
  const std::vector<ScriptStep> halfScript = {leave(0.1), aboveS(2.0), aboveS(4.2), crossDown(4.5)};
  const std::vector<ScriptStep> flatScript = {leave(0.1), aboveS(2.0), aboveS(4.2), crossFlat(4.5)};

  struct Case {
    const char* name;
    bool accept;
    std::string reason;
  };
  const std::vector<Case> cases = {
      {"counter: departure up, one counted downward crossing in the window", true,
       counterVerdict(-1, 1, I(4.3, 4.4), 0, true, {depart(0, 0.1), above(0.1, 2.0), above(2.0, 4.2), touch(-0.5, -0.3, 4.2, 4.5)})},
      {"counter: same section, window on the 2nd counted crossing with returns=1", false,
       counterVerdict(1, 1, I(17.3, 17.5), 0, true, twoUp)},
      {"counter: same section, window on the 2nd counted crossing with returns=2", true,
       counterVerdict(1, 2, I(17.3, 17.5), 0, true, twoUp)},
      {"counter: same section, window on the 1st counted crossing with returns=1", true,
       counterVerdict(1, 1, I(8.6, 8.8), 0, true, twoUp)},
      {"counter: touching step where ds/dt may vanish", false,
       counterVerdict(-1, 1, I(1.0, 1.2), 1, false, {above(0, 1.0), touch(-0.2, 0.1, 1.0, 1.2)})},
      {"counter: ds/dt changes sign inside one touching run", false,
       counterVerdict(-1, 1, I(1.0, 1.4), 1, false, {above(0, 1.0), touch(-0.5, -0.3, 1.0, 1.2), touch(0.3, 0.5, 1.2, 1.4)})},
      {"counter: counted crossing spread over several touching steps", true,
       counterVerdict(-1, 1, I(1.0, 1.6), 1, false,
                      {above(0, 1.0), touch(-0.5, -0.3, 1.0, 1.2), touch(-0.5, -0.3, 1.2, 1.4), touch(-0.5, -0.3, 1.4, 1.6)})},
      {"counter: start straddles the section moving in the counted direction", false,
       counterVerdict(-1, 1, I(13.0, 13.1), 0, false,
                      {touch(-0.5, -0.3, 0, 0.1), below(0.1, 6.0), touch(0.3, 0.5, 6.0, 6.2), above(6.2, 12.9), touch(-0.5, -0.3, 12.9, 13.2)})},
      {"counter: start straddles the section moving against the counted direction", true,
       counterVerdict(1, 1, I(6.0, 6.2), 0, false, {touch(-0.5, -0.3, 0, 0.1), below(0.1, 6.0), touch(0.3, 0.5, 6.0, 6.2)})},
      {"counter: the run in the window crosses in the wrong direction", false,
       counterVerdict(-1, 1, I(1.0, 1.2), -1, false, {below(0, 1.0), touch(0.3, 0.5, 1.0, 1.2)})},
      {"counter: window overlaps the departure from the section", false,
       counterVerdict(1, 1, I(0.0, 0.05), 0, true, {depart(0, 0.1), above(0.1, 0.2)})},
      {"counter: no touching run overlaps the window", false, counterVerdict(-1, 1, I(5.0, 6.0), 1, false, {above(0, 7.0)})},
      {"counter: two touching runs overlap the window", false,
       counterVerdict(-1, 1, I(1.0, 2.2), 1, false,
                      {above(0, 1.0), touch(-0.5, -0.3, 1.0, 1.2), above(1.2, 1.8), touch(-0.5, -0.3, 1.8, 2.0), below(2.0, 2.5)})},
      {"counter: sign change across a run disagrees with ds/dt", false,
       counterVerdict(-1, 1, I(5.0, 5.2), 1, false, {above(0, 1.0), touch(0.3, 0.5, 1.0, 1.2), below(1.2, 5.3)})},
      {"counter: departure from the section returns to the starting side", false,
       counterVerdict(1, 1, I(9.0, 10.0), 0, true, {depart(0, 0.1), below(0.1, 0.2)})},
      {"counter: start before the section, the immediate counted crossing is returned", true,
       counterVerdict(-1, 1, I(0.0, 0.1), 1, false, {touch(-0.5, -0.3, 0, 0.1)})},
      {"counter: start before the section, returns=2 counts the immediate crossing", true,
       counterVerdict(-1, 2, I(13.0, 13.2), 1, false,
                      {touch(-0.5, -0.3, 0, 0.1), below(0.1, 6.0), touch(0.3, 0.5, 6.0, 6.2), above(6.2, 12.9), touch(-0.5, -0.3, 12.9, 13.2)})},
      {"engine c0: same section, CAPD window on the 2nd counted crossing with returns=1", false,
       engineVerdict("c0", 1, 1, I(17.3, 17.5), twoUpScript)},
      {"engine c0: same section, same steps with returns=2", true, engineVerdict("c0", 1, 2, I(17.3, 17.5), twoUpScript)},
      {"engine c1: touching tube step where ds/dt may vanish", false, engineVerdict("c1", -1, 1, I(4.3, 4.4), flatScript)},
      {"engine c1: one counted crossing in the window", true, engineVerdict("c1", -1, 1, I(4.3, 4.4), halfScript)},
      {"engine c0: affine image from a different return window than the hull", false,
       engineVerdict("c0", -1, 1, I(4.3, 4.4), halfScript, I(4.3, 4.45))},
      {"report: departure run then one counted crossing", true,
       counterReport(-1, 1, I(4.3, 4.4), 0, true, {depart(0, 0.1), above(0.1, 2.0), above(2.0, 4.2), touch(-0.5, -0.3, 4.2, 4.5)},
                     1, 0, 1, I(4.2, 4.5))},
      {"report: returns=2 on the same section counts one crossing before the window", true,
       counterReport(1, 2, I(17.3, 17.5), 0, true, twoUp, 1, 1, 4, I(17.2, 17.6))},
      {"report: crossing spread over several touching steps is one window run", true,
       counterReport(-1, 1, I(1.0, 1.6), 1, false,
                     {above(0, 1.0), touch(-0.5, -0.3, 1.0, 1.2), touch(-0.5, -0.3, 1.2, 1.4), touch(-0.5, -0.3, 1.4, 1.6)},
                     1, 0, 0, I(1.0, 1.6))},
      {"engine c0 output: crossing lines follow the tube lines", true,
       engineReport("c0", -1, 1, I(4.3, 4.4), halfScript, 1, 0, 1, 4.2, 4.5)},
      {"engine c1 output: crossing lines follow the tube lines", true,
       engineReport("c1", -1, 1, I(4.3, 4.4), halfScript, 1, 0, 1, 4.2, 4.5)},
      {"engine c0 output: returns=2 reports the crossing counted before the window", true,
       engineReport("c0", 1, 2, I(17.3, 17.5), twoUpScript, 1, 1, 4, 17.2, 17.6)},
  };
  int failed = 0;
  for (const Case& c : cases) {
    const bool accepted = c.reason.empty();
    const bool ok = accepted == c.accept;
    failed += !ok;
    std::printf("%s %s%s%s\n", ok ? "ok" : "FAIL", c.name, accepted ? "" : " | ", c.reason.c_str());
  }
  std::printf("%d/%zu passed\n", static_cast<int>(cases.size()) - failed, cases.size());
  return failed ? 1 : 0;
}
