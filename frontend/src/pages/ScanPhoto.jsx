"use client"

import { useNavigate } from "react-router-dom"
import { useState, useEffect } from "react"
import axios from "axios"
import lettuceMascot from "../assets/lettuce.jpeg"
import { homeStyles } from "../styles/pageStyles"

const ScanPhoto = () => {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultMedia, setResultMedia] = useState(null)
  const [summary, setSummary] = useState({})
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [hoveredButton, setHoveredButton] = useState(null)

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) setSidebarOpen(true)
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const onFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultMedia(null)
      setSummary({})
    }
  }

  const onUpload = async () => {
    if (!selectedFile) return
    setLoading(true)
    const formData = new FormData()
    formData.append("file", selectedFile)
    try {
      const res = await axios.post("http://127.0.0.1:8000/detect", formData)
      setResultMedia(res.data.image)
      setSummary(res.data.summary)
    } catch (err) {
      alert("Gagal memproses deteksi.")
    } finally {
      setLoading(false)
    }
  }

  const getImageSrc = (media) => {
    if (!media) return ""
    return media.startsWith("data:") ? media : `data:image/jpeg;base64,${media}`
  }

  const getDetectionStatus = () => {
    const totalDiseases = Object.values(summary).reduce((a, b) => a + b, 0)
    const diseaseTypes = Object.keys(summary)
    
    // Jika hanya ada "healthy" atau tidak ada deteksi
    if (totalDiseases === 0 || (diseaseTypes.length === 1 && diseaseTypes[0].toLowerCase() === "healthy")) {
      return { 
        status: "✓ Healthy", 
        color: "#10b981", 
        bg: "rgba(16, 185, 129, 0.1)",
        description: "Tanaman dalam kondisi sehat, tidak ada penyakit terdeteksi"
      }
    }
    
    // Hitung hanya penyakit yang sebenarnya (exclude "healthy")
    const actualDiseases = diseaseTypes.filter(d => d.toLowerCase() !== "healthy")
    const actualDiseaseCount = actualDiseases.reduce((sum, d) => sum + (summary[d] || 0), 0)
    
    if (actualDiseaseCount === 0) {
      return { 
        status: "✓ Healthy", 
        color: "#10b981", 
        bg: "rgba(16, 185, 129, 0.1)",
        description: "Tanaman dalam kondisi sehat"
      }
    } else if (actualDiseaseCount <= 2 && actualDiseases.length === 1) {
      return { 
        status: "⚠️ Low Risk", 
        color: "#3b82f6", 
        bg: "rgba(59, 130, 246, 0.1)",
        description: "Risiko penyakit rendah, monitoring rutin disarankan"
      }
    } else if (actualDiseaseCount <= 5 && actualDiseases.length <= 2) {
      return { 
        status: "⚠️ Medium Risk", 
        color: "#f59e0b", 
        bg: "rgba(245, 158, 11, 0.1)",
        description: "Risiko penyakit sedang, lakukan treatment segera"
      }
    } else {
      return { 
        status: "🚨 High Risk", 
        color: "#ef4444", 
        bg: "rgba(239, 68, 68, 0.1)",
        description: "Risiko penyakit tinggi - segera ambil tindakan penyelamatan"
      }
    }
  }

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
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "600", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>🏠</span>
            <span>Dashboard</span>
          </button>

          <button onClick={() => { navigate("/scan-realtime"); setSidebarOpen(false) }}
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s ease", background: "transparent", color: "#cbd5e1" }}>
            <span style={{ fontSize: "18px" }}>📷</span>
            <span>Real-time Scan</span>
          </button>

          <button
            style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", fontSize: "14px", fontWeight: "600", cursor: "pointer", transition: "all 0.2s ease", background: "linear-gradient(135deg, rgba(0, 188, 212, 0.15) 0%, rgba(0, 188, 212, 0.05) 100%)", border: "1px solid rgba(0,188,212,0.3)", color: "#00bcd4" }}>
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
            Photo Disease Detection
          </h1>
          <p
            style={{
              fontSize: window.innerWidth < 768 ? "14px" : "16px",
              color: "#64748b",
              fontWeight: "500",
            }}
          >
            Upload lettuce photos for AI-powered disease analysis
          </p>
        </div>

        {!previewUrl && !resultMedia && (
          <>
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
                    ✓ Photo Analysis
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
                Upload Your Lettuce Photo
              </h3>

              <div
                style={{
                  padding: window.innerWidth < 768 ? "20px" : "28px",
                  background: "#ffffff",
                  border: hoveredButton === "upload" ? "2px solid #3b82f6" : "2px solid #e2e8f0",
                  borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  textAlign: "center",
                  boxShadow:
                    hoveredButton === "upload"
                      ? "0 12px 32px rgba(59, 130, 246, 0.15)"
                      : "0 2px 8px rgba(0, 0, 0, 0.04)",
                  transform: hoveredButton === "upload" ? "translateY(-4px)" : "translateY(0)",
                  maxWidth: "600px",
                  margin: "0 auto",
                }}
                onMouseEnter={() => setHoveredButton("upload")}
                onMouseLeave={() => setHoveredButton(null)}
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
                    margin: "0 auto 16px auto",
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
                  Upload Lettuce Photo
                </h4>
                <p
                  style={{
                    fontSize: window.innerWidth < 768 ? "13px" : "14px",
                    color: "#64748b",
                    lineHeight: "1.5",
                    marginBottom: "16px",
                  }}
                >
                  Select a clear photo of lettuce plants for disease detection and analysis
                </p>
                <input
                  type="file"
                  onChange={onFileChange}
                  accept="image/*"
                  style={{ display: "none" }}
                  id="fileInput"
                />
                <label htmlFor="fileInput" style={{
                  display: "inline-block",
                  padding: "12px 32px",
                  background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
                  color: "#ffffff",
                  borderRadius: "10px",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  boxShadow: "0 4px 12px rgba(59, 130, 246, 0.3)",
                  border: "none",
                }}>
                  Choose File
                </label>
              </div>
            </div>
          </>
        )}

        {previewUrl && !resultMedia && (
          <div
            style={{
              background: "#ffffff",
              borderRadius: window.innerWidth < 768 ? "16px" : "24px",
              padding: window.innerWidth < 768 ? "20px" : "32px",
              boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
              maxWidth: "900px",
              margin: "0 auto",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "24px",
              }}
            >
              <h3
                style={{
                  margin: 0,
                  fontSize: window.innerWidth < 768 ? "18px" : "20px",
                  fontWeight: "700",
                  color: "#1e293b",
                }}
              >
                Image Preview
              </h3>
              <button
                onClick={() => {
                  setPreviewUrl(null)
                  setSelectedFile(null)
                }}
                style={{
                  background: "#fee2e2",
                  border: "none",
                  padding: "10px 20px",
                  borderRadius: "10px",
                  cursor: "pointer",
                  color: "#dc2626",
                  fontSize: "13px",
                  fontWeight: "600",
                  transition: "all 0.2s ease",
                }}
              >
                ✕ Reset
              </button>
            </div>

            <div
              style={{
                borderRadius: "16px",
                overflow: "hidden",
                marginBottom: "20px",
                boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
                background: "#f8fafc",
              }}
            >
              <img src={previewUrl || "/placeholder.svg"} alt="Preview" style={{ width: "100%", height: "auto", display: "block" }} />
            </div>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "16px",
                background: "#f8fafc",
                borderRadius: "12px",
                marginBottom: "24px",
              }}
            >
              <span style={{ fontSize: "24px" }}>📄</span>
              <span
                style={{
                  fontSize: "14px",
                  color: "#475569",
                  fontWeight: "500",
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {selectedFile?.name}
              </span>
            </div>

            <button
              onClick={onUpload}
              disabled={loading}
              style={{
                width: "100%",
                padding: "16px",
                background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                color: "#ffffff",
                border: "none",
                borderRadius: "12px",
                fontSize: "16px",
                fontWeight: "700",
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s ease",
                boxShadow: "0 8px 20px rgba(16, 185, 129, 0.3)",
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? "⏳ Analyzing..." : "🔍 Start Detection"}
            </button>
          </div>
        )}

        {resultMedia && (
          <>
            <div
              style={{
                background: "#ffffff",
                borderRadius: window.innerWidth < 768 ? "16px" : "24px",
                padding: window.innerWidth < 768 ? "20px" : "32px",
                boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
                marginBottom: "24px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "24px",
                }}
              >
                <h2
                  style={{
                    margin: 0,
                    fontSize: window.innerWidth < 768 ? "20px" : "24px",
                    fontWeight: "800",
                    color: "#1e293b",
                  }}
                >
                  Detection Result
                </h2>
                <button
                  onClick={() => {
                    setResultMedia(null)
                    setPreviewUrl(null)
                    setSelectedFile(null)
                    setSummary({})
                  }}
                  style={{
                    background: "#ef4444",
                    border: "none",
                    padding: "10px 20px",
                    borderRadius: "10px",
                    cursor: "pointer",
                    color: "#ffffff",
                    fontSize: "14px",
                    fontWeight: "600",
                    transition: "all 0.2s ease",
                  }}
                >
                  🔄 New Scan
                </button>
              </div>

              <div
                style={{
                  borderRadius: "16px",
                  overflow: "hidden",
                  boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
                  marginBottom: "24px",
                  border: "3px solid #10b981",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "12px 20px",
                    background: "#ffffff",
                    borderBottom: "1px solid #e2e8f0",
                  }}
                >
                  <span
                    style={{
                      fontSize: "11px",
                      fontWeight: "800",
                      color: "#1e293b",
                      letterSpacing: "1px",
                    }}
                  >
                    DETECTED IMAGE
                  </span>
                  <span style={{ fontSize: "11px", fontWeight: "700", color: "#10b981" }}>● ANALYZED</span>
                </div>
                <div
                  style={{
                    padding: "20px",
                    background: "#1e293b",
                    display: "flex",
                    justifyContent: "center",
                    minHeight: "400px",
                  }}
                >
                  <img
                    src={getImageSrc(resultMedia) || "/placeholder.svg"}
                    alt="Detection Result"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "500px",
                      borderRadius: "8px",
                    }}
                  />
                </div>
              </div>

              <div
                style={{
                  display: "flex",
                  gap: "12px",
                  justifyContent: "center",
                  flexWrap: "wrap",
                }}
              >
                <button
                  style={{
                    padding: "12px 24px",
                    background: "#3b82f6",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "10px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  📥 Download Report
                </button>
                <a
                  href={getImageSrc(resultMedia)}
                  download="lettuce_detection.jpg"
                  style={{
                    padding: "12px 24px",
                    background: "#10b981",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "10px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    textDecoration: "none",
                    display: "inline-block",
                  }}
                >
                  🖼️ Export Image
                </a>
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  window.innerWidth < 768
                    ? "1fr"
                    : "repeat(auto-fit, minmax(300px, 1fr))",
                gap: "24px",
              }}
            >
              <div
                style={{
                  background: "#ffffff",
                  borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                  padding: "24px",
                  boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    background: "#1e293b",
                    color: "#ffffff",
                    padding: "10px 16px",
                    borderRadius: "10px",
                    fontSize: "11px",
                    fontWeight: "700",
                    marginBottom: "20px",
                    letterSpacing: "1px",
                  }}
                >
                  <span style={{ fontSize: "16px" }}>📊</span>
                  DETECTION SUMMARY
                </div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "16px",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "16px",
                      background: "#f8fafc",
                      borderRadius: "12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#64748b",
                        fontWeight: "600",
                      }}
                    >
                      Total Detected:
                    </span>
                    <div
                      style={{
                        fontSize: "32px",
                        fontWeight: "800",
                        color: "#1e293b",
                      }}
                    >
                      {Object.values(summary).reduce((a, b) => a + b, 0)}
                    </div>
                  </div>

                  {Object.keys(summary).length > 0 ? (
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "10px",
                      }}
                    >
                      {Object.entries(summary).map(([disease, count]) => (
                        <div
                          key={disease}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "12px",
                            background: "#f8fafc",
                            borderRadius: "8px",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "14px",
                              color: "#475569",
                              fontWeight: "500",
                            }}
                          >
                            {disease}
                          </span>
                          <span
                            style={{
                              fontSize: "16px",
                              fontWeight: "700",
                              color: "#1e293b",
                              padding: "4px 12px",
                              background: "#e2e8f0",
                              borderRadius: "6px",
                            }}
                          >
                            {count}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p
                      style={{
                        fontSize: "14px",
                        color: "#94a3b8",
                        textAlign: "center",
                        padding: "20px",
                      }}
                    >
                      No diseases detected
                    </p>
                  )}
                </div>
              </div>

              <div
                style={{
                  background: "#ffffff",
                  borderRadius: window.innerWidth < 768 ? "16px" : "20px",
                  padding: "24px",
                  boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    background: "#1e293b",
                    color: "#ffffff",
                    padding: "10px 16px",
                    borderRadius: "10px",
                    fontSize: "11px",
                    fontWeight: "700",
                    marginBottom: "20px",
                    letterSpacing: "1px",
                  }}
                >
                  <span style={{ fontSize: "16px" }}>📋</span>
                  DETECTION LOG
                </div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "16px",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "12px 0",
                      borderBottom: "1px solid #e2e8f0",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#64748b",
                        fontWeight: "500",
                      }}
                    >
                      Analysis Type:
                    </span>
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#1e293b",
                        fontWeight: "600",
                      }}
                    >
                      Photo Detection
                    </span>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "12px 0",
                      borderBottom: "1px solid #e2e8f0",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#64748b",
                        fontWeight: "500",
                      }}
                    >
                      Status:
                    </span>
                    <span
                      style={{
                        fontSize: "13px",
                        fontWeight: "600",
                        color: getDetectionStatus().color,
                        padding: "4px 12px",
                        background: getDetectionStatus().bg,
                        borderRadius: "6px",
                      }}
                    >
                      {getDetectionStatus().status}
                    </span>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "12px 0",
                      borderBottom: "1px solid #e2e8f0",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#64748b",
                        fontWeight: "500",
                      }}
                    >
                      Model:
                    </span>
                    <span
                      style={{
                        fontSize: "14px",
                        color: "#1e293b",
                        fontWeight: "600",
                      }}
                    >
                      AI Disease Detection
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: "12px",
                      color: "#94a3b8",
                      padding: "12px 0",
                      borderTop: "1px solid #e2e8f0",
                    }}
                  >
                    Scan Date: {new Date().toLocaleString("id-ID")}
                  </div>
                  <div
                    style={{
                      marginTop: "16px",
                      padding: "12px",
                      background: "rgba(0,0,0,0.02)",
                      borderRadius: "8px",
                      borderLeft: "4px solid " + getDetectionStatus().color,
                    }}
                  >
                    <div style={{ fontSize: "12px", fontWeight: "700", color: getDetectionStatus().color, marginBottom: "4px" }}>
                      💡 Insight
                    </div>
                    <div style={{ fontSize: "12px", color: "#64748b", lineHeight: "1.5" }}>
                      {getDetectionStatus().description}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default ScanPhoto

