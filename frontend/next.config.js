/** @type {import('next').NextConfig} */
const nextConfig = {
  // API 프록시: 개발 시 CORS 없이 FastAPI 호출
  async rewrites() {
    const dest = process.env.API_URL ?? "http://localhost:8000";
    return [
      { source: "/health",       destination: `${dest}/health` },
      { source: "/api/:path*",   destination: `${dest}/api/:path*` },
    ];
  },
};

module.exports = nextConfig;
