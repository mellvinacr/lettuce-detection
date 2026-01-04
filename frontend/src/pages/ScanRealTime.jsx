"use client"

import { useRef, useState, useEffect, useCallback } from "react"
import Webcam from "react-webcam"
import axios from "axios"
import { useNavigate } from "react-router-dom"
// Sidebar component inlined (reverted) to restore previous layout
import { scanRealtimeStyles } from "../styles/pageStyles"

const ScanRealTime = () => {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const navigate = useNavigate()
  const [prediction, setPrediction] = useState({})
  const [isScanning, setIsScanning] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [hoveredButton, setHoveredButton] = useState(null)

  const drawBoxes = (detections) => {
    const canvas = canvasRef.current
    const video = webcamRef.current?.video
    if (!canvas || !video) return
    const dpr = window.devicePixelRatio || 1
    const displayW = video.clientWidth
    const displayH = video.clientHeight

    // set internal canvas size for crisp lines on high-dpi and match CSS size
    canvas.width = Math.round(displayW * dpr)
    canvas.height = Math.round(displayH * dpr)
    canvas.style.width = `${displayW}px`
    canvas.style.height = `${displayH}px`

    const ctx = canvas.getContext("2d")
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // Clear canvas for each new frame (use logical pixels)
    ctx.clearRect(0, 0, displayW, displayH)

    // Model input size we send to backend
    const inputSize = 300
    const scaleX = displayW / inputSize
    const scaleY = displayH / inputSize

    detections.forEach((det) => {
      let [x1, y1, x2, y2] = det.box.map(Number)
      const isHealthy = det.label.toLowerCase() === "healthy"

      // Scale from model/input coords (300x300) to displayed video size
      x1 = x1 * scaleX
      y1 = y1 * scaleY
      x2 = x2 * scaleX
      y2 = y2 * scaleY

      // Clamp to video bounds
      x1 = Math.max(0, Math.min(x1, displayW))
      y1 = Math.max(0, Math.min(y1, displayH))
      x2 = Math.max(0, Math.min(x2, displayW))
      y2 = Math.max(0, Math.min(y2, displayH))

      // Draw bounding box
      ctx.strokeStyle = isHealthy ? "#00FF00" : "#FF0000"
      ctx.lineWidth = 3
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      // Draw label background
      ctx.fillStyle = isHealthy ? "#00FF00" : "#FF0000"
      ctx.font = "bold 16px Arial"
      const text = `${det.label} (${(det.score * 100).toFixed(0)}%)`
      const textMetrics = ctx.measureText(text)
      const textX = x1
      const textY = y1 > 25 ? y1 - 25 : y1
      ctx.fillRect(textX, textY, textMetrics.width + 10, 20)

      // Draw label text
      ctx.fillStyle = "#000000"
      ctx.fillText(text, textX + 5, textY + 15)
    })
  }

  // Letterbox helper: take a dataURL or HTMLImageElement and produce a letterboxed Blob
  const letterboxDataUrlToBlob = (dataUrl, target = 300) => {
    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement("canvas")
        canvas.width = target
        canvas.height = target
        const ctx = canvas.getContext("2d")

        // fill background (black)
        ctx.fillStyle = "#000"
        ctx.fillRect(0, 0, target, target)

        const iw = img.width
        const ih = img.height
        const scale = Math.min(target / iw, target / ih)
        const nw = Math.round(iw * scale)
        const nh = Math.round(ih * scale)
        const dx = Math.round((target - nw) / 2)
        const dy = Math.round((target - nh) / 2)

        ctx.drawImage(img, 0, 0, iw, ih, dx, dy, nw, nh)

        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92)
      }
      img.src = dataUrl
    })
  }

  const capture = useCallback(async () => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) return

      // Create letterboxed 300x300 blob before sending so backend sees consistent input
      const letterboxBlob = await letterboxDataUrlToBlob(imageSrc, 300)
      const formData = new FormData()
      formData.append("file", letterboxBlob, "scan.jpg")

      try {
        const res = await axios.post("http://127.0.0.1:8000/detect-realtime", formData)
        setPrediction(res.data.summary)
        console.log("[FRONTEND] Detections:", res.data.detections)
        console.log("[FRONTEND] Summary:", res.data.summary)

        // Draw bounding boxes if detections exist
        if (res.data.detections) {
          drawBoxes(res.data.detections)
        }
      } catch (err) {
        console.error("[FRONTEND] Detection error:", err)
      }
    }
  }, [])

  useEffect(() => {
    let interval
    if (isScanning) {
      interval = setInterval(capture, 1000) // Detect every 1 second
    } else {
      // Clear boxes when scanning stops
      const ctx = canvasRef.current?.getContext("2d")
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    return () => clearInterval(interval)
  }, [isScanning, capture])

  return (
    <div style={scanRealtimeStyles.container}>
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

          <button onClick={() => { navigate("/"); setSidebarOpen(false) }} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "600", cursor: "pointer", transition: "all 0.2s ease", background: "linear-gradient(135deg, rgba(0, 188, 212, 0.15) 0%, rgba(0, 188, 212, 0.05) 100%)", border: "1px solid rgba(0,188,212,0.3)", color: "#00bcd4" }}>
            <span style={{ fontSize: "18px" }}>🏠</span>
            <span>Dashboard</span>
          </button>

          <button onClick={() => { navigate("/scan-realtime"); setSidebarOpen(false) }} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>📷</span>
            <span>Real-time Scan</span>
          </button>

          <button onClick={() => { navigate("/photo"); setSidebarOpen(false) }} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>🖼️</span>
            <span>Photo Upload</span>
          </button>

          <button onClick={() => { navigate("/video"); setSidebarOpen(false) }} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
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
          ...styles.mainContent,
          paddingLeft: window.innerWidth < 1024 ? "20px" : "40px",
        }}
      >
        {window.innerWidth < 1024 && (
          <button onClick={() => setSidebarOpen(!sidebarOpen)} style={scanRealtimeStyles.mobileMenuBtn}>
            ☰
          </button>
        )}

        <div
          style={{
            marginBottom: window.innerWidth < 768 ? "24px" : "40px",
            marginTop: window.innerWidth < 1024 ? "60px" : "0",
          }}
        >
          <h1 style={scanRealtimeStyles.pageTitle}>Real-time Disease Detection</h1>
          <p style={scanRealtimeStyles.pageSubtitle}>Live webcam monitoring with AI-powered disease detection and bounding boxes</p>
        </div>

        <div style={scanRealtimeStyles.heroCard}>
          <div style={{ flex: 1 }}>
            <div style={scanRealtimeStyles.aiPoweredBadge}>
              <span style={scanRealtimeStyles.badgeText}>REAL-TIME AI DETECTION</span>
            </div>
            <h2 style={scanRealtimeStyles.heroTitle}>Pemantauan Langsung dengan Computer Vision</h2>
            <p style={scanRealtimeStyles.heroDescription}>
              Deteksi penyakit secara real-time menggunakan webcam dengan visualisasi bounding box untuk setiap tanaman
              yang terdeteksi. Sistem akan melakukan analisis setiap detik untuk hasil yang akurat.
            </p>
            <div style={scanRealtimeStyles.featureBadges}>
              <div style={scanRealtimeStyles.featureBadge1}>✓ Live Detection</div>
              <div style={scanRealtimeStyles.featureBadge2}>✓ Bounding Box</div>
            </div>
          </div>

          <div style={scanRealtimeStyles.mascotWrapper}>
            <div style={scanRealtimeStyles.mascotIcon}>🥬</div>
          </div>
        </div>

        <div style={scanRealtimeStyles.methodSection}>
          <h3 style={scanRealtimeStyles.sectionTitle}>Live Detection Feed</h3>

          <div
            style={{
              display: window.innerWidth < 1024 ? "block" : "grid",
              gridTemplateColumns: window.innerWidth < 1024 ? "1fr" : "2fr 1fr",
              gap: "24px",
              marginBottom: "32px",
            }}
          >
            <div style={scanRealtimeStyles.videoCard}>
              <div style={scanRealtimeStyles.cardHeader}>
                <span style={{ fontSize: "18px" }}>📹</span>
                <span>Visual Detection Feed</span>
              </div>
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  backgroundColor: "#000",
                  borderRadius: "12px",
                  overflow: "hidden",
                }}
              >
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  width={640}
                  height={480}
                  style={{ width: "100%", display: "block" }}
                />
                {/* Canvas overlay for drawing bounding boxes */}
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={480}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                  }}
                />
              </div>
              <button
                onClick={() => setIsScanning(!isScanning)}
                style={{
                  ...styles.actionButton,
                  background: isScanning
                    ? "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                    : "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "translateY(-2px)"
                  e.currentTarget.style.boxShadow = isScanning
                    ? "0 8px 24px rgba(239, 68, 68, 0.3)"
                    : "0 8px 24px rgba(16, 185, 129, 0.3)"
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0)"
                  e.currentTarget.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.1)"
                }}
              >
                <span style={{ fontSize: "18px" }}>{isScanning ? "⏹" : "▶"}</span>
                <span>{isScanning ? "Hentikan Pemindaian" : "Mulai Deteksi Real-Time"}</span>
              </button>
            </div>

            {/* Detection log summary */}
            <div style={scanRealtimeStyles.summaryCard}>
              <div style={scanRealtimeStyles.cardHeader}>
                <span style={{ fontSize: "18px" }}>📊</span>
                <span>Detection Log Summary</span>
              </div>
              <div style={scanRealtimeStyles.summaryContent}>
                {Object.keys(prediction).length > 0 ? (
                  <table style={scanRealtimeStyles.summaryTable}>
                    <thead>
                      <tr>
                        <th style={scanRealtimeStyles.tableHeader}>Classification</th>
                        <th style={scanRealtimeStyles.tableHeader}>Qty</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(prediction).map(([name, count]) => (
                        <tr key={name} style={scanRealtimeStyles.tableRow}>
                          <td style={scanRealtimeStyles.tableCell}>
                            <span
                              style={{
                                ...styles.badge,
                                background:
                                  name.toLowerCase() === "healthy"
                                    ? "linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05))"
                                    : "linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05))",
                                color: name.toLowerCase() === "healthy" ? "#065f46" : "#991b1b",
                                border:
                                  name.toLowerCase() === "healthy"
                                    ? "1px solid rgba(16, 185, 129, 0.3)"
                                    : "1px solid rgba(239, 68, 68, 0.3)",
                              }}
                            >
                              {name.toUpperCase()}
                            </span>
                          </td>
                          <td style={{ ...styles.tableCell, fontWeight: "700", color: "#1e293b" }}>{count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div style={scanRealtimeStyles.emptyState}>
                    <div style={{ fontSize: "48px", marginBottom: "16px" }}>📷</div>
                    <p style={{ color: "#94a3b8", fontSize: "14px", fontWeight: "500" }}>
                      Menunggu deteksi...
                      <br />
                      Klik tombol untuk memulai
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Info section */}
          <div style={scanRealtimeStyles.infoCard}>
            <div style={scanRealtimeStyles.infoIcon}>💡</div>
            <div>
              <div style={scanRealtimeStyles.infoTitle}>Cara Menggunakan Real-time Detection</div>
              <div style={scanRealtimeStyles.infoText}>
                Klik tombol "Mulai Deteksi Real-Time" untuk memulai pemantauan. Sistem akan menganalisis feed webcam
                setiap detik dan menampilkan bounding box di sekitar tanaman yang terdeteksi. Kotak hijau menandakan
                tanaman sehat, sedangkan kotak merah menandakan penyakit terdeteksi.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    background: "#f1f5f9",
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif',
    position: "relative",
    width: "100vw",
    overflowX: "hidden",
    boxSizing: "border-box",
  },
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.5)",
    zIndex: 40,
    display: window.innerWidth >= 1024 ? "none" : "block",
  },
  sidebar: {
    width: "260px",
    background: "linear-gradient(180deg, #1e293b 0%, #0f172a 100%)",
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    boxShadow: "4px 0 24px rgba(0, 0, 0, 0.1)",
    top: 0,
    bottom: 0,
    zIndex: 50,
    transition: "left 0.3s ease",
    overflowY: "auto",
  },
  sidebarBrand: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "16px",
    background: "rgba(255, 255, 255, 0.05)",
    borderRadius: "12px",
    marginBottom: "24px",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  },
  brandIcon: {
    width: "40px",
    height: "40px",
    borderRadius: "10px",
    background: "linear-gradient(135deg, #00bcd4 0%, #0097a7 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "20px",
    boxShadow: "0 4px 12px rgba(0, 188, 212, 0.3)",
  },
  brandName: {
    color: "#ffffff",
    fontSize: "16px",
    fontWeight: "700",
    letterSpacing: "0.3px",
  },
  brandSub: {
    color: "#64748b",
    fontSize: "12px",
    fontWeight: "500",
  },
  navGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  navLabel: {
    color: "#94a3b8",
    fontSize: "11px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "1px",
    padding: "8px 12px",
    marginTop: "8px",
  },
  navButton: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px 16px",
    background: "transparent",
    border: "1px solid transparent",
    borderRadius: "10px",
    color: "#cbd5e1",
    fontSize: "14px",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  navButtonActive: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px 16px",
    background: "linear-gradient(135deg, rgba(0, 188, 212, 0.15) 0%, rgba(0, 188, 212, 0.05) 100%)",
    border: "1px solid rgba(0, 188, 212, 0.3)",
    borderRadius: "10px",
    color: "#00bcd4",
    fontSize: "14px",
    fontWeight: "600",
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  sidebarFooter: {
    marginTop: "auto",
    padding: "16px",
    background: "rgba(255, 255, 255, 0.03)",
    borderRadius: "10px",
    border: "1px solid rgba(255, 255, 255, 0.05)",
  },
  statusIndicator: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  statusLabel: {
    color: "#64748b",
    fontSize: "11px",
    fontWeight: "600",
  },
  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#10b981",
    boxShadow: "0 0 8px rgba(16, 185, 129, 0.5)",
  },
  statusText: {
    color: "#94a3b8",
    fontSize: "12px",
    fontWeight: "500",
  },
  mainContent: {
    flex: 1,
    padding: window.innerWidth < 768 ? "20px" : window.innerWidth < 1024 ? "24px" : "40px",
    paddingRight: window.innerWidth < 768 ? "20px" : window.innerWidth < 1024 ? "24px" : "40px",
    overflowY: "auto",
    width: "100%",
    maxWidth: "100%",
    boxSizing: "border-box",
  },
  mobileMenuBtn: {
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
    color: "#fff",
  },
  pageTitle: {
    fontSize: window.innerWidth < 768 ? "24px" : "32px",
    fontWeight: "800",
    color: "#1e293b",
    marginBottom: "8px",
    letterSpacing: "-0.02em",
  },
  pageSubtitle: {
    fontSize: window.innerWidth < 768 ? "14px" : "16px",
    color: "#64748b",
    fontWeight: "500",
  },
  heroCard: {
    background: "linear-gradient(135deg, rgba(0, 188, 212, 0.08) 0%, rgba(15, 23, 42, 0.03) 100%)",
    borderRadius: window.innerWidth < 768 ? "16px" : "24px",
    padding: window.innerWidth < 768 ? "24px" : "40px",
    marginBottom: window.innerWidth < 768 ? "24px" : "32px",
    boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
    display: window.innerWidth < 768 ? "block" : "flex",
    gap: "40px",
    alignItems: "center",
    border: "1px solid rgba(0, 188, 212, 0.15)",
    width: "100%",
  },
  aiPoweredBadge: {
    display: "inline-block",
    padding: "8px 16px",
    background: "linear-gradient(135deg, rgba(0, 188, 212, 0.1) 0%, rgba(0, 188, 212, 0.05) 100%)",
    borderRadius: "100px",
    marginBottom: "16px",
    border: "1px solid rgba(0, 188, 212, 0.2)",
  },
  badgeText: {
    fontSize: window.innerWidth < 768 ? "11px" : "13px",
    fontWeight: "700",
    color: "#00838f",
    letterSpacing: "0.5px",
  },
  heroTitle: {
    fontSize: window.innerWidth < 768 ? "20px" : window.innerWidth < 1024 ? "24px" : "28px",
    fontWeight: "800",
    color: "#0f172a",
    marginBottom: "12px",
    lineHeight: "1.3",
  },
  heroDescription: {
    fontSize: window.innerWidth < 768 ? "14px" : "16px",
    color: "#64748b",
    lineHeight: "1.6",
    marginBottom: "24px",
  },
  featureBadges: {
    display: "flex",
    flexWrap: "wrap",
    gap: "12px",
  },
  featureBadge1: {
    padding: window.innerWidth < 768 ? "10px 16px" : "12px 20px",
    background: "rgba(16, 185, 129, 0.1)",
    borderRadius: "12px",
    fontSize: window.innerWidth < 768 ? "12px" : "13px",
    fontWeight: "600",
    color: "#065f46",
  },
  featureBadge2: {
    padding: window.innerWidth < 768 ? "10px 16px" : "12px 20px",
    background: "rgba(59, 130, 246, 0.1)",
    borderRadius: "12px",
    fontSize: window.innerWidth < 768 ? "12px" : "13px",
    fontWeight: "600",
    color: "#1e3a8a",
  },
  mascotWrapper: {
    width: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "200px" : "240px",
    height: window.innerWidth < 768 ? "200px" : window.innerWidth < 1024 ? "200px" : "240px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  mascotIcon: {
    fontSize: window.innerWidth < 768 ? "120px" : "140px",
    filter: "drop-shadow(0 8px 16px rgba(0, 0, 0, 0.1))",
  },
  methodSection: {
    marginBottom: window.innerWidth < 768 ? "24px" : "32px",
    width: "100%",
  },
  sectionTitle: {
    fontSize: window.innerWidth < 768 ? "16px" : "18px",
    fontWeight: "700",
    color: "#1e293b",
    marginBottom: "16px",
  },
  videoCard: {
    background: "rgba(255, 255, 255, 0.5)",
    backdropFilter: "blur(10px)",
    borderRadius: "16px",
    padding: "24px",
    boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
    border: "1px solid rgba(255, 255, 255, 0.8)",
    marginBottom: window.innerWidth < 1024 ? "24px" : "0",
  },
  summaryCard: {
    background: "rgba(255, 255, 255, 0.5)",
    backdropFilter: "blur(10px)",
    borderRadius: "16px",
    padding: "24px",
    boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
    border: "1px solid rgba(255, 255, 255, 0.8)",
  },
  cardHeader: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    fontSize: "16px",
    fontWeight: "700",
    color: "#1e293b",
    marginBottom: "16px",
    paddingBottom: "12px",
    borderBottom: "2px solid rgba(0, 188, 212, 0.2)",
  },
  actionButton: {
    width: "100%",
    padding: "14px 24px",
    borderRadius: "12px",
    border: "none",
    color: "#ffffff",
    fontSize: "15px",
    fontWeight: "600",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    transition: "all 0.2s ease",
    marginTop: "16px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
  },
  summaryContent: {
    minHeight: "300px",
    display: "flex",
    flexDirection: "column",
  },
  summaryTable: {
    width: "100%",
    borderCollapse: "separate",
    borderSpacing: "0 8px",
  },
  tableHeader: {
    textAlign: "left",
    padding: "12px",
    fontSize: "13px",
    fontWeight: "700",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  tableRow: {
    background: "rgba(255, 255, 255, 0.6)",
    transition: "all 0.2s ease",
  },
  tableCell: {
    padding: "16px 12px",
    fontSize: "14px",
    color: "#475569",
  },
  badge: {
    padding: "6px 12px",
    borderRadius: "8px",
    fontSize: "12px",
    fontWeight: "700",
    display: "inline-block",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    minHeight: "300px",
  },
  infoCard: {
    background: "linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.03) 100%)",
    borderRadius: "16px",
    padding: "24px",
    display: "flex",
    gap: "16px",
    alignItems: "flex-start",
    border: "1px solid rgba(59, 130, 246, 0.15)",
  },
  infoIcon: {
    fontSize: "32px",
    flexShrink: 0,
  },
  infoTitle: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#1e293b",
    marginBottom: "8px",
  },
  infoText: {
    fontSize: "14px",
    color: "#64748b",
    lineHeight: "1.6",
  },
}

export default ScanRealTime

