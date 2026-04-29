import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-bg text-textPri relative overflow-hidden selection:bg-primaryBlue selection:text-white">
      {/* 백그라운드 메쉬 그라디언트 */}
      <div className="absolute inset-0 bg-mesh opacity-80 pointer-events-none z-0" />

      {/* 네비게이션 바 */}
      <nav className="fixed top-0 w-full glass border-b border-white/5 px-8 py-4 z-50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="Shoealls Logo" className="h-10 w-auto" />
        </div>
        <div className="flex items-center gap-6">
          <Link href="#features" className="text-textSec hover:text-white transition-colors text-[14px] font-medium hidden sm:block">
            핵심 기술
          </Link>
          <Link href="#architecture" className="text-textSec hover:text-white transition-colors text-[14px] font-medium hidden sm:block">
            AI 추론 구조
          </Link>
          <Link
            href="/dashboard"
            className="px-6 py-2.5 font-bold text-white bg-primaryBlue rounded-xl transition-all hover:bg-primaryBlue/90 hover:scale-105 shadow-md"
          >
            대시보드 접속
          </Link>
          <ThemeToggle />
        </div>
      </nav>

      <main className="relative z-10 pt-32 pb-20 px-8 max-w-7xl mx-auto flex flex-col items-center">
        
        {/* Hero Section */}
        <section className="flex flex-col items-center text-center mt-20 mb-32 animate-fade-in-up">
          <div className="inline-block mb-6 px-4 py-1.5 rounded-full border border-primaryBlue/20 bg-primaryBlue/5 backdrop-blur-md">
            <span className="text-primaryBlue text-[12px] font-bold tracking-widest uppercase">Next-Gen Gait Analysis</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-black tracking-tight mb-8 leading-tight">
            당신의 걸음걸이가 <br/>
            <span className="text-primaryBlue">
              미래의 건강을 예측합니다
            </span>
          </h1>
          <p className="text-textSec text-lg md:text-xl max-w-2xl font-light mb-12 leading-relaxed">
            멀티모달 센서 융합과 딥러닝 기반의 Chain-of-Reasoning을 통해<br className="hidden md:block" />
            보행 패턴을 분류하고 잠재적인 질환과 부상 위험을 조기에 발견하세요.
          </p>
          <div className="flex gap-4">
            <Link
              href="/dashboard"
              className="px-8 py-4 rounded-2xl bg-primaryBlue text-white font-bold text-lg hover:bg-primaryBlue/90 transition-all hover:-translate-y-1 shadow-md"
            >
              대시보드 시작하기
            </Link>
            <Link
              href="#features"
              className="px-8 py-4 rounded-2xl glass-card text-textPri font-medium text-lg hover:bg-black/5 dark:hover:bg-white/5 transition-all hover:-translate-y-1"
            >
              기술 알아보기
            </Link>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="w-full mt-20 mb-32">
          <div className="text-center mb-16 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            <h2 className="text-3xl font-bold mb-4">멀티모달 AI 분석 엔진</h2>
            <p className="text-textSec">IMU, 족저압, 스켈레톤 데이터를 융합하여 정밀한 분석을 제공합니다.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: "👟",
                title: "보행 패턴 분류",
                desc: "파킨슨, 운동실조, 절뚝거림 등 비정상적인 보행 패턴을 실시간으로 감지하고 분류합니다.",
                color: "primaryBlue"
              },
              {
                icon: "⚕️",
                title: "질환 위험 예측",
                desc: "신경계, 근골격계 등 14개 질환에 대한 위험도를 머신러닝 앙상블로 진단합니다.",
                color: "medicalTeal"
              },
              {
                icon: "⚠️",
                title: "역학적 부상 경고",
                desc: "보행 비대칭 및 충격 하중을 분석하여 부위별(무릎, 허리, 발목 등) 부상 위험을 경고합니다.",
                color: "warningAmber"
              }
            ].map((feature, i) => {
              const colorClasses = {
                primaryBlue: {
                  card: "border-primaryBlue/20 hover:border-primaryBlue/50",
                  iconBg: "bg-primaryBlue/10 border-primaryBlue/20 text-primaryBlue",
                },
                medicalTeal: {
                  card: "border-medicalTeal/20 hover:border-medicalTeal/50",
                  iconBg: "bg-medicalTeal/10 border-medicalTeal/20 text-medicalTeal",
                },
                warningAmber: {
                  card: "border-warningAmber/20 hover:border-warningAmber/50",
                  iconBg: "bg-warningAmber/10 border-warningAmber/20 text-warningAmber",
                }
              }[feature.color as "primaryBlue" | "medicalTeal" | "warningAmber"];

              return (
                <div 
                  key={i} 
                  className={`glass-card p-8 rounded-3xl border-t transition-all duration-300 group hover:-translate-y-2 ${colorClasses.card}`}
                >
                  <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform border ${colorClasses.iconBg}`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-textPri mb-3 tracking-wide">{feature.title}</h3>
                  <p className="text-textSec leading-relaxed text-[15px]">{feature.desc}</p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Call To Action */}
        <section className="w-full relative py-20 rounded-[40px] overflow-hidden glass-card border-primaryBlue/20 text-center mb-20 bg-primaryBlue/5">
          <div className="relative z-10">
            <h2 className="text-4xl font-bold mb-6 text-textPri tracking-tight">당신의 걸음걸이를 지금 분석해보세요</h2>
            <p className="text-textSec mb-10 max-w-lg mx-auto text-lg">
              슈올즈 AI 플랫폼은 데모 모드를 지원하여 센서 데이터 없이도<br/>
              강력한 멀티모달 분석을 체험할 수 있습니다.
            </p>
            <Link
              href="/dashboard"
              className="inline-flex px-10 py-4 bg-primaryBlue text-white font-bold text-lg rounded-full hover:bg-primaryBlue/90 transition-all hover:scale-105 shadow-md"
            >
              무료로 체험하기
            </Link>
          </div>
        </section>
      </main>

      <footer className="glass border-t border-white/5 py-10 px-8 text-center relative z-20 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="Shoealls Logo" className="h-6 w-auto grayscale opacity-70" />
          </div>
          <div className="text-textMuted text-[13px]">
            © 2026 Shoealls Advanced Gait AI. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
