"use client"

import { useNavigate } from "react-router-dom"
import { useState } from "react"
// Sidebar component inlined (reverted) to restore previous layout
import lettuceMascot from "../assets/lettuce.jpeg"
import { homeStyles } from "../styles/pageStyles"

const Home = () => {
  const navigate = useNavigate()
  const [hoveredCard, setHoveredCard] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div style={homeStyles.container}>
      {/* Inlined sidebar (restored) */}
      {sidebarOpen && window.innerWidth < 1024 && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0, 0, 0, 0.5)",
            zIndex: 40,
            display: window.innerWidth >= 1024 ? "none" : "block",
          }}
        />
      )}

      <div
        style={{
          width: "260px",
          background: "linear-gradient(180deg, #1e293b 0%, #0f172a 100%)",
          padding: "24px",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          boxShadow: "4px 0 24px rgba(0, 0, 0, 0.1)",
          position: window.innerWidth < 1024 ? "fixed" : "relative",
          left: window.innerWidth < 1024 ? (sidebarOpen ? "0" : "-260px") : "auto",
          top: 0,
          bottom: 0,
          zIndex: 50,
          transition: "left 0.3s ease",
          overflowY: "auto",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px", background: "rgba(255,255,255,0.05)", borderRadius: "12px", marginBottom: "24px", border: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ width: "40px", height: "40px", borderRadius: "10px", background: "linear-gradient(135deg, #00bcd4 0%, #0097a7 100%)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "20px", boxShadow: "0 4px 12px rgba(0,188,212,0.3)" }}>🥬</div>
          <div>
            <div style={{ color: "#ffffff", fontSize: "16px", fontWeight: "700", letterSpacing: "0.3px" }}>LettuceEye AI</div>
            <div style={{ color: "#64748b", fontSize: "12px", fontWeight: "500" }}>Disease Detection</div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          <div style={{ color: "#94a3b8", fontSize: "11px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "1px", padding: "8px 12px", marginTop: "8px" }}>MAIN PAGES</div>

          <button onClick={() => { navigate("/"); setSidebarOpen(false) }}
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "600", cursor: "pointer", transition: "all 0.2s ease", background: "linear-gradient(135deg, rgba(0, 188, 212, 0.15) 0%, rgba(0, 188, 212, 0.05) 100%)", border: "1px solid rgba(0,188,212,0.3)", color: "#00bcd4" }}>
            <span style={{ fontSize: "18px" }}>🏠</span>
            <span>Dashboard</span>
          </button>

          <button onClick={() => { navigate("/scan-realtime"); setSidebarOpen(false) }}
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>📷</span>
            <span>Real-time Scan</span>
          </button>

          <button onClick={() => { navigate("/photo"); setSidebarOpen(false) }}
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>🖼️</span>
            <span>Photo Upload</span>
          </button>

          <button onClick={() => { navigate("/video"); setSidebarOpen(false) }}
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>🎥</span>
            <span>Video Analysis</span>
          </button>
        </div>

        <div style={{ marginTop: "auto", padding: "16px", background: "rgba(255,255,255,0.03)", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <div style={{ color: "#64748b", fontSize: "11px", fontWeight: "600" }}>System Status</div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px rgba(16,185,129,0.5)" }} />
              <span style={{ color: "#94a3b8", fontSize: "12px", fontWeight: "500" }}>All systems operational</span>
            </div>
          </div>
        </div>
      </div>

      <div
        style={{
          flex: 1,
          padding: window.innerWidth < 768 ? "20px" : window.innerWidth < 1024 ? "24px" : "40px",
          overflowY: "auto",
          width: "100%",
          maxWidth: "100%",
          boxSizing: "border-box",
        }}
      >
        {window.innerWidth < 1024 && (
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            style={{
              position: "fixed",
              top: "20px",
              left: "20px",
              zIndex: 30,
              width: "48px",
              height: "48px",
              borderRadius: "12px",
              background: "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
              border: "none",
              boxShadow: "0 4px 16px rgba(0, 0, 0, 0.2)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "24px",
            }}
          >
            ☰
          </button>
        )}

        <div
          style={{
            marginBottom: window.innerWidth < 768 ? "24px" : "40px",
            marginTop: window.innerWidth < 1024 ? "60px" : "0",
          }}
        >
          <h1
            style={{
              fontSize: window.innerWidth < 768 ? "24px" : "32px",
              fontWeight: "800",
              color: "#1e293b",
              marginBottom: "8px",
              letterSpacing: "-0.02em",
            }}
          >
            Deteksi Penyakit Selada Hijau
          </h1>
          <p
            style={{
              fontSize: window.innerWidth < 768 ? "14px" : "16px",
              color: "#64748b",
              fontWeight: "500",
            }}
          >
            Pilih metode deteksi untuk menganalisis kesehatan tanaman selada Anda
          </p>
        </div>

        <div
          style={{
            background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
            borderRadius: window.innerWidth < 768 ? "16px" : "24px",
            padding: window.innerWidth < 768 ? "24px" : window.innerWidth < 1024 ? "32px" : "48px",
            marginBottom: window.innerWidth < 768 ? "24px" : "32px",
            boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
            border: "1px solid rgba(226, 232, 240, 0.8)",
            display: "flex",
            flexDirection: window.innerWidth < 1024 ? "column" : "row",
            alignItems: "center",
            justifyContent: "space-between",
            gap: window.innerWidth < 768 ? "24px" : "48px",
          }}
        >
          <div style={{ flex: 1 }}>
            <div
              style={{
                display: "inline-block",
                padding: "8px 16px",
                background: "linear-gradient(135deg, rgba(0, 188, 212, 0.1) 0%, rgba(0, 188, 212, 0.05) 100%)",
                borderRadius: "100px",
                marginBottom: "16px",
                border: "1px solid rgba(0, 188, 212, 0.2)",
              }}
            >
              <span
                style={{
                  fontSize: window.innerWidth < 768 ? "11px" : "13px",
                  fontWeight: "700",
                  color: "#00838f",
                  letterSpacing: "0.5px",
                }}
              >
                AI-POWERED DETECTION
              </span>
            </div>
            <h2
              style={{
                fontSize: window.innerWidth < 768 ? "20px" : window.innerWidth < 1024 ? "24px" : "28px",
                fontWeight: "800",
                color: "#0f172a",
                marginBottom: "12px",
                lineHeight: "1.3",
              }}
            >
              Teknologi Computer Vision untuk Pertanian Modern
            </h2>
            <p
              style={{
                fontSize: window.innerWidth < 768 ? "14px" : "16px",
                color: "#64748b",
                lineHeight: "1.6",
                marginBottom: "24px",
              }}
            >
              Deteksi dini penyakit pada tanaman selada menggunakan deep learning dan computer vision untuk meningkatkan
              hasil panen dan kualitas tanaman.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
              <div
                style={{
                  padding: window.innerWidth < 768 ? "10px 16px" : "12px 20px",
                  background: "rgba(16, 185, 129, 0.1)",
                  borderRadius: "12px",
                  fontSize: window.innerWidth < 768 ? "12px" : "13px",
                  fontWeight: "600",
                  color: "#065f46",
                }}
              >
                ✓ Akurasi Tinggi
              </div>
              <div
                style={{
                  padding: window.innerWidth < 768 ? "10px 16px" : "12px 20px",
                  background: "rgba(59, 130, 246, 0.1)",
                  borderRadius: "12px",
                  fontSize: window.innerWidth < 768 ? "12px" : "13px",
                  fontWeight: "600",
                  color: "#1e3a8a",
                }}
              >
                ✓ Real-time Analysis
              </div>
            </div>
          </div>

          <div
            style={{
              width: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "240px" : "280px",
              height: window.innerWidth < 768 ? "240px" : window.innerWidth < 1024 ? "240px" : "280px",
              borderRadius: window.innerWidth < 768 ? "16px" : "24px",
              overflow: "hidden",
              boxShadow: "0 20px 40px rgba(0, 188, 212, 0.2)",
              border: "4px solid #ffffff",
              position: "relative",
              flexShrink: 0,
            }}
          >
            <img
              src={lettuceMascot || "/placeholder.svg"}
              alt="Lettuce Plant"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
          </div>
        </div>

        <div style={{ marginBottom: "24px" }}>
          <h3
            style={{
              fontSize: window.innerWidth < 768 ? "16px" : "18px",
              fontWeight: "700",
              color: "#1e293b",
              marginBottom: "20px",
            }}
          >
            Metode Deteksi
          </h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                window.innerWidth < 768
                  ? "1fr"
                  : window.innerWidth < 1024
                    ? "repeat(2, 1fr)"
                    : "repeat(auto-fit, minmax(300px, 1fr))",
              gap: window.innerWidth < 768 ? "16px" : "20px",
            }}
          >
            <button
              onClick={() => navigate("/scan-realtime")}
              onMouseEnter={() => setHoveredCard("realtime")}
              onMouseLeave={() => setHoveredCard(null)}
              style={{
                padding: window.innerWidth < 768 ? "20px" : "28px",
                background: "#ffffff",
                border: hoveredCard === "realtime" ? "2px solid #10b981" : "2px solid #e2e8f0",
                borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                cursor: "pointer",
                transition: "all 0.3s ease",
                textAlign: "left",
                boxShadow:
                  hoveredCard === "realtime" ? "0 12px 32px rgba(16, 185, 129, 0.15)" : "0 2px 8px rgba(0, 0, 0, 0.04)",
                transform: hoveredCard === "realtime" ? "translateY(-4px)" : "translateY(0)",
              }}
            >
              <div
                style={{
                  width: window.innerWidth < 768 ? "48px" : "56px",
                  height: window.innerWidth < 768 ? "48px" : "56px",
                  borderRadius: window.innerWidth < 768 ? "12px" : "16px",
                  background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: window.innerWidth < 768 ? "24px" : "28px",
                  marginBottom: "16px",
                  boxShadow: "0 8px 20px rgba(16, 185, 129, 0.3)",
                }}
              >
                📷
              </div>
              <h4
                style={{
                  fontSize: window.innerWidth < 768 ? "16px" : "18px",
                  fontWeight: "700",
                  color: "#0f172a",
                  marginBottom: "8px",
                }}
              >
                Scan Real-time
              </h4>
              <p
                style={{
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  color: "#64748b",
                  lineHeight: "1.5",
                  marginBottom: "16px",
                }}
              >
                Deteksi penyakit langsung menggunakan kamera untuk analisis instant
              </p>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  fontWeight: "600",
                  color: "#10b981",
                }}
              >
                <span>Mulai Scan</span>
                <span>→</span>
              </div>
            </button>

            <button
              onClick={() => navigate("/photo")}
              onMouseEnter={() => setHoveredCard("photo")}
              onMouseLeave={() => setHoveredCard(null)}
              style={{
                padding: window.innerWidth < 768 ? "20px" : "28px",
                background: "#ffffff",
                border: hoveredCard === "photo" ? "2px solid #3b82f6" : "2px solid #e2e8f0",
                borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                cursor: "pointer",
                transition: "all 0.3s ease",
                textAlign: "left",
                boxShadow:
                  hoveredCard === "photo" ? "0 12px 32px rgba(59, 130, 246, 0.15)" : "0 2px 8px rgba(0, 0, 0, 0.04)",
                transform: hoveredCard === "photo" ? "translateY(-4px)" : "translateY(0)",
              }}
            >
              <div
                style={{
                  width: window.innerWidth < 768 ? "48px" : "56px",
                  height: window.innerWidth < 768 ? "48px" : "56px",
                  borderRadius: window.innerWidth < 768 ? "12px" : "16px",
                  background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: window.innerWidth < 768 ? "24px" : "28px",
                  marginBottom: "16px",
                  boxShadow: "0 8px 20px rgba(59, 130, 246, 0.3)",
                }}
              >
                🖼️
              </div>
              <h4
                style={{
                  fontSize: window.innerWidth < 768 ? "16px" : "18px",
                  fontWeight: "700",
                  color: "#0f172a",
                  marginBottom: "8px",
                }}
              >
                Upload Photo
              </h4>
              <p
                style={{
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  color: "#64748b",
                  lineHeight: "1.5",
                  marginBottom: "16px",
                }}
              >
                Upload foto tanaman untuk mendapatkan hasil analisis mendalam
              </p>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  fontWeight: "600",
                  color: "#3b82f6",
                }}
              >
                <span>Upload Foto</span>
                <span>→</span>
              </div>
            </button>

            <button
              onClick={() => navigate("/video")}
              onMouseEnter={() => setHoveredCard("video")}
              onMouseLeave={() => setHoveredCard(null)}
              style={{
                padding: window.innerWidth < 768 ? "20px" : "28px",
                background: "#ffffff",
                border: hoveredCard === "video" ? "2px solid #a855f7" : "2px solid #e2e8f0",
                borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                cursor: "pointer",
                transition: "all 0.3s ease",
                textAlign: "left",
                boxShadow:
                  hoveredCard === "video" ? "0 12px 32px rgba(168, 85, 247, 0.15)" : "0 2px 8px rgba(0, 0, 0, 0.04)",
                transform: hoveredCard === "video" ? "translateY(-4px)" : "translateY(0)",
              }}
            >
              <div
                style={{
                  width: window.innerWidth < 768 ? "48px" : "56px",
                  height: window.innerWidth < 768 ? "48px" : "56px",
                  borderRadius: window.innerWidth < 768 ? "12px" : "16px",
                  background: "linear-gradient(135deg, #a855f7 0%, #9333ea 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: window.innerWidth < 768 ? "24px" : "28px",
                  marginBottom: "16px",
                  boxShadow: "0 8px 20px rgba(168, 85, 247, 0.3)",
                }}
              >
                🎥
              </div>
              <h4
                style={{
                  fontSize: window.innerWidth < 768 ? "16px" : "18px",
                  fontWeight: "700",
                  color: "#0f172a",
                  marginBottom: "8px",
                }}
              >
                Analisis Video
              </h4>
              <p
                style={{
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  color: "#64748b",
                  lineHeight: "1.5",
                  marginBottom: "16px",
                }}
              >
                Proses video untuk deteksi komprehensif pada multiple frames
              </p>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: window.innerWidth < 768 ? "13px" : "14px",
                  fontWeight: "600",
                  color: "#a855f7",
                }}
              >
                <span>Proses Video</span>
                <span>→</span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home
