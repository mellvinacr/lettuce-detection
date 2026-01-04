"use client"

import { useState, useEffect, useRef } from "react"
import axios from "axios"
import { useNavigate } from "react-router-dom"
// Sidebar component inlined (reverted) to restore previous layout
import { scanVideoStyles } from "../styles/pageStyles"

const ScanVideo = () => {
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultMedia, setResultMedia] = useState(null)
  const resultMediaRef = useRef(null)
  const [summary, setSummary] = useState({})
  const [loading, setLoading] = useState(false)

  // Cleanup URL untuk mencegah memory leak
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      if (resultMedia && resultMedia.startsWith("blob:")) URL.revokeObjectURL(resultMedia)
    }
  }, [previewUrl, resultMedia])

  const onFileChange = (e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith("video")) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResultMedia(null)
      setSummary({})
    } else {
      alert("Mohon pilih file video!")
    }
  }

  const onUpload = async () => {
    if (!selectedFile) return
    setLoading(true)

    // Try to extract a first-frame thumbnail and send it as a letterboxed 300x300 'thumb' blob
    const formData = new FormData()
    formData.append("file", selectedFile)
    try {
      const thumbBlob = await extractFirstFrameLetterbox(selectedFile, 300)
      if (thumbBlob) formData.append("thumb", thumbBlob, "thumb.jpg")
    } catch (e) {
      console.warn("Could not create thumbnail for video upload:", e)
    }

    try {
      const res = await axios.post("http://127.0.0.1:8000/detect-video", formData, {
        responseType: "arraybuffer",
      })

      console.log("[SCANVIDEO] Response headers:", res.headers)
      // Create a proper Blob with explicit MIME type (some servers omit it)
      const blob = new Blob([res.data], { type: "video/mp4" })
      console.log("[SCANVIDEO] Blob size:", blob.size)
      const videoUrl = URL.createObjectURL(blob)
      // revoke previous blob URL if exists
      if (resultMediaRef.current) {
        try { URL.revokeObjectURL(resultMediaRef.current) } catch (e) {}
      }
      resultMediaRef.current = videoUrl
      setResultMedia(videoUrl)

      const summaryHeader = res.headers["x-summary"]
      if (summaryHeader) {
        setSummary(JSON.parse(summaryHeader))
      }
    } catch (err) {
      console.error(err)
      alert("Gagal memproses video.")
    } finally {
      setLoading(false)
    }
  }

  // Extract first frame from video file and return a letterboxed Blob (300x300)
  const extractFirstFrameLetterbox = (file, target = 300) => {
    return new Promise((resolve, reject) => {
      try {
        const url = URL.createObjectURL(file)
        const vid = document.createElement("video")
        vid.preload = "metadata"
        vid.src = url
        vid.muted = true
        vid.playsInline = true
        vid.onloadeddata = () => {
          // draw the first frame
          const iw = vid.videoWidth
          const ih = vid.videoHeight
          const scale = Math.min(target / iw, target / ih)
          const nw = Math.round(iw * scale)
          const nh = Math.round(ih * scale)
          const dx = Math.round((target - nw) / 2)
          const dy = Math.round((target - nh) / 2)

          const canvas = document.createElement("canvas")
          canvas.width = target
          canvas.height = target
          const ctx = canvas.getContext("2d")
          ctx.fillStyle = "#000"
          ctx.fillRect(0, 0, target, target)
          ctx.drawImage(vid, 0, 0, iw, ih, dx, dy, nw, nh)

          canvas.toBlob((blob) => {
            URL.revokeObjectURL(url)
            resolve(blob)
          }, "image/jpeg", 0.92)
        }
        vid.onerror = (e) => {
          URL.revokeObjectURL(url)
          reject(e)
        }
      } catch (err) {
        reject(err)
      }
    })
  }

  const resetUpload = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    if (resultMediaRef.current) {
      try { URL.revokeObjectURL(resultMediaRef.current) } catch (e) {}
      resultMediaRef.current = null
    }
    setResultMedia(null)
    setSummary({})
  }

  return (
    <div style={scanVideoStyles.container}>
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

      {/* Main Content */}
      <div style={scanVideoStyles.mainContent}>
        {window.innerWidth < 1024 && (
          <button onClick={() => setSidebarOpen(true)} style={scanVideoStyles.mobileMenuBtn}>
            ☰
          </button>
        )}

        <div style={{ marginBottom: window.innerWidth < 768 ? "24px" : "32px" }}>
          <h1 style={scanVideoStyles.pageTitle}>Video Disease Detection</h1>
          <p style={scanVideoStyles.pageSubtitle}>Analisis video selada menggunakan AI-powered detection</p>
        </div>

        {/* Hero Card */}
        <div style={scanVideoStyles.heroCard}>
          <div style={{ flex: 1 }}>
            <div style={scanVideoStyles.aiPoweredBadge}>
              <span style={scanVideoStyles.badgeText}>AI-POWERED DETECTION</span>
            </div>
            <h2 style={scanVideoStyles.heroTitle}>Teknologi Computer Vision untuk Pertanian Modern</h2>
            <p style={scanVideoStyles.heroDescription}>
              Deteksi dini penyakit pada tanaman selada menggunakan deep learning dan computer vision untuk meningkatkan
              hasil panen dan kualitas tanaman.
            </p>
            <div style={scanVideoStyles.featureBadges}>
              <div style={scanVideoStyles.featureBadge2}>✓ Video Analysis</div>
              <div style={scanVideoStyles.featureBadge1}>✓ Multi-Frame Detection</div>
            </div>
          </div>
          <div style={scanVideoStyles.mascotWrapper}>
            <div style={scanVideoStyles.mascotIcon}>🥬</div>
          </div>
        </div>

        {/* Upload Section */}
        {!selectedFile && (
          <div style={scanVideoStyles.methodSection}>
            <h3 style={scanVideoStyles.sectionTitle}>Metode Deteksi</h3>
            <div style={scanVideoStyles.uploadCard}>
              <div style={scanVideoStyles.uploadIconBox}>
                <div style={scanVideoStyles.uploadIcon}>🎥</div>
              </div>
              <h4 style={scanVideoStyles.uploadTitle}>Upload Lettuce Video</h4>
              <p style={scanVideoStyles.uploadDescription}>
                Select a clear video of lettuce plants for disease detection and analysis
              </p>
              <label style={scanVideoStyles.chooseFileBtn}>
                Choose File
                <input type="file" onChange={onFileChange} accept="video/*" style={{ display: "none" }} />
              </label>
            </div>
          </div>
        )}

        {/* Preview Section */}
        {selectedFile && !resultMedia && (
          <div style={scanVideoStyles.previewContainer}>
            <div style={scanVideoStyles.previewHeader}>
              <h3 style={scanVideoStyles.previewTitle}>Video Preview</h3>
              <button onClick={resetUpload} style={scanVideoStyles.resetButton}>
                ✕ Reset
              </button>
            </div>
            <div style={scanVideoStyles.previewImageContainer}>
              <video src={previewUrl} controls style={{ width: "100%", display: "block" }} />
            </div>
            <div style={scanVideoStyles.fileInfoBox}>
              <span style={scanVideoStyles.fileIcon}>🎬</span>
              <span style={scanVideoStyles.fileName}>{selectedFile.name}</span>
            </div>
            <button
              onClick={onUpload}
              disabled={loading}
              style={{
                ...styles.detectButton,
                opacity: loading ? 0.6 : 1,
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "● AI is Scanning Frames..." : "● Start Detection"}
            </button>
          </div>
        )}

        {/* Result Section */}
        {resultMedia && (
          <div style={scanVideoStyles.resultContainer}>
            <div style={scanVideoStyles.resultHeader}>
              <h3 style={scanVideoStyles.resultTitle}>Detection Result</h3>
              <button onClick={resetUpload} style={scanVideoStyles.newScanButton}>
                ✕ New Scan
              </button>
            </div>

            <div style={scanVideoStyles.detectedImageCard}>
              <div style={scanVideoStyles.imageCardHeader}>
                <span style={scanVideoStyles.imageLabel}>DETECTED VIDEO</span>
                <span style={scanVideoStyles.analyzedBadge}>● ANALYZED</span>
              </div>
              <div style={scanVideoStyles.imageCardBody}>
                <video
                  key={resultMedia}
                  src={resultMedia}
                  controls
                  autoPlay
                  style={{ width: "100%", maxHeight: "500px", borderRadius: "8px" }}
                />
              </div>
            </div>

            <div style={scanVideoStyles.actionButtonsContainer}>
              <a href={resultMedia} download="detected_lettuce.mp4" style={scanVideoStyles.exportButton}>
                📥 Download Video Hasil
              </a>
            </div>
          </div>
        )}

        {/* Summary Section */}
        {Object.keys(summary).length > 0 && (
          <div style={scanVideoStyles.summaryCardsWrapper}>
            <div style={scanVideoStyles.summaryCard}>
              <div style={scanVideoStyles.statsHeader}>
                <span style={scanVideoStyles.statsIcon}>📊</span>
                <span>DETECTION SUMMARY</span>
              </div>
              <div style={scanVideoStyles.diseaseTable}>
                <div style={scanVideoStyles.tableHeader}>
                  <div style={scanVideoStyles.tableHeaderCell}>Classification</div>
                  <div style={scanVideoStyles.tableHeaderCell}>Instances</div>
                </div>
                {Object.entries(summary).map(([name, count]) => (
                  <div key={name} style={scanVideoStyles.tableRow}>
                    <div style={scanVideoStyles.tableCellDisease}>{name}</div>
                    <div style={scanVideoStyles.tableCellCount}>{count} Occurrence(s)</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={scanVideoStyles.summaryCard}>
              <div style={scanVideoStyles.logCard}>
                <div style={scanVideoStyles.logHeader}>
                  <h4 style={scanVideoStyles.logTitle}>Detection Log</h4>
                  <span style={{ ...styles.statusBadge, backgroundColor: "#dcfce7", color: "#15803d" }}>SUCCESS</span>
                </div>
                <div style={scanVideoStyles.logDetails}>
                  <p style={scanVideoStyles.logText}>
                    <strong>File:</strong> {selectedFile?.name}
                  </p>
                  <p style={scanVideoStyles.logText}>
                    <strong>Status:</strong> Analysis Complete
                  </p>
                  <div style={scanVideoStyles.timestampBox}>
                    <span style={scanVideoStyles.timestampLabel}>TIMESTAMP</span>
                    <span style={scanVideoStyles.timestampValue}>{new Date().toLocaleString("id-ID")}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
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
    background: "rgba(168, 85, 247, 0.1)",
    borderRadius: "12px",
    fontSize: window.innerWidth < 768 ? "12px" : "13px",
    fontWeight: "600",
    color: "#6b21a8",
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
    marginBottom: "20px",
  },
  uploadCard: {
    background: "rgba(255, 255, 255, 0.4)",
    backdropFilter: "blur(10px)",
    borderRadius: window.innerWidth < 768 ? "16px" : "24px",
    padding: window.innerWidth < 768 ? "32px 20px" : "48px 40px",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.08)",
    border: "2px dashed rgba(168, 85, 247, 0.3)",
    textAlign: "center",
    transition: "all 0.3s ease",
    maxWidth: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "600px" : "800px",
    margin: "0 auto",
    width: "100%",
  },
  uploadIconBox: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "16px",
  },
  uploadIcon: {
    width: window.innerWidth < 768 ? "48px" : "56px",
    height: window.innerWidth < 768 ? "48px" : "56px",
    borderRadius: window.innerWidth < 768 ? "12px" : "16px",
    background: "linear-gradient(135deg, #a855f7 0%, #9333ea 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: window.innerWidth < 768 ? "24px" : "28px",
    boxShadow: "0 8px 20px rgba(168, 85, 247, 0.3)",
  },
  uploadTitle: {
    fontSize: window.innerWidth < 768 ? "16px" : "18px",
    fontWeight: "700",
    color: "#0f172a",
    marginBottom: "8px",
  },
  uploadDescription: {
    fontSize: window.innerWidth < 768 ? "13px" : "14px",
    color: "#64748b",
    lineHeight: "1.5",
    marginBottom: "16px",
  },
  chooseFileBtn: {
    display: "inline-block",
    padding: "12px 32px",
    background: "linear-gradient(135deg, #a855f7 0%, #9333ea 100%)",
    color: "#ffffff",
    borderRadius: "10px",
    fontSize: "14px",
    fontWeight: "600",
    cursor: "pointer",
    transition: "all 0.2s ease",
    boxShadow: "0 4px 12px rgba(168, 85, 247, 0.3)",
    border: "none",
  },
  previewContainer: {
    background: "rgba(255, 255, 255, 0.5)",
    backdropFilter: "blur(10px)",
    borderRadius: window.innerWidth < 768 ? "16px" : "24px",
    padding: window.innerWidth < 768 ? "20px" : "32px",
    boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
    maxWidth: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "720px" : "1200px",
    margin: "0 auto",
    width: "100%",
  },
  previewHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "24px",
  },
  previewTitle: {
    margin: 0,
    fontSize: window.innerWidth < 768 ? "18px" : "20px",
    fontWeight: "700",
    color: "#1e293b",
  },
  resetButton: {
    background: "#fee2e2",
    border: "none",
    padding: "10px 20px",
    borderRadius: "10px",
    cursor: "pointer",
    color: "#dc2626",
    fontSize: "13px",
    fontWeight: "600",
    transition: "all 0.2s ease",
  },
  previewImageContainer: {
    borderRadius: "16px",
    overflow: "hidden",
    marginBottom: "20px",
    boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
    background: "#f8fafc",
  },
  fileInfoBox: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "16px",
    background: "#f8fafc",
    borderRadius: "12px",
    marginBottom: "24px",
  },
  fileIcon: {
    fontSize: "24px",
  },
  fileName: {
    fontSize: "14px",
    color: "#475569",
    fontWeight: "500",
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  detectButton: {
    width: "100%",
    padding: "16px",
    background: "linear-gradient(135deg, #a855f7 0%, #9333ea 100%)",
    color: "#ffffff",
    border: "none",
    borderRadius: "12px",
    fontSize: "16px",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.2s ease",
    boxShadow: "0 8px 20px rgba(168, 85, 247, 0.3)",
  },
  resultContainer: {
    background: "#ffffff",
    borderRadius: window.innerWidth < 768 ? "16px" : "24px",
    padding: window.innerWidth < 768 ? "20px" : "32px",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.06)",
    marginBottom: "24px",
    maxWidth: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "720px" : "1200px",
    margin: "0 auto 24px auto",
    width: "100%",
  },
  resultHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "24px",
  },
  resultTitle: {
    margin: 0,
    fontSize: window.innerWidth < 768 ? "20px" : "24px",
    fontWeight: "800",
    color: "#1e293b",
  },
  newScanButton: {
    background: "#ef4444",
    border: "none",
    padding: "10px 20px",
    borderRadius: "10px",
    cursor: "pointer",
    color: "#ffffff",
    fontSize: "14px",
    fontWeight: "600",
    transition: "all 0.2s ease",
  },
  detectedImageCard: {
    borderRadius: "16px",
    overflow: "hidden",
    boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)",
    marginBottom: "24px",
    border: "3px solid #a855f7",
  },
  imageCardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "12px 20px",
    background: "#ffffff",
    borderBottom: "1px solid #e2e8f0",
  },
  imageLabel: {
    fontSize: "11px",
    fontWeight: "800",
    color: "#1e293b",
    letterSpacing: "1px",
  },
  analyzedBadge: {
    fontSize: "11px",
    fontWeight: "700",
    color: "#a855f7",
  },
  imageCardBody: {
    padding: "20px",
    background: "#1e293b",
    display: "flex",
    justifyContent: "center",
    minHeight: "400px",
  },
  actionButtonsContainer: {
    display: "flex",
    gap: "12px",
    justifyContent: "center",
    flexWrap: "wrap",
  },
  exportButton: {
    padding: "12px 24px",
    background: "#a855f7",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    fontSize: "14px",
    fontWeight: "600",
    cursor: "pointer",
    transition: "all 0.2s ease",
    textDecoration: "none",
    display: "inline-block",
  },
  summaryCardsWrapper: {
    display: "grid",
    gridTemplateColumns: window.innerWidth < 768 ? "1fr" : window.innerWidth < 1024 ? "1fr" : "1fr 1fr",
    gap: "24px",
    maxWidth: window.innerWidth < 768 ? "100%" : window.innerWidth < 1024 ? "720px" : "1200px",
    margin: "0 auto",
    width: "100%",
  },
  summaryCard: {
    background: "#ffffff",
    borderRadius: window.innerWidth < 768 ? "16px" : "20px",
    padding: window.innerWidth < 768 ? "20px" : "28px",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.06)",
    border: "1px solid #e2e8f0",
    width: "100%",
  },
  statsHeader: {
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
  },
  statsIcon: {
    fontSize: "16px",
  },
  diseaseTable: {
    marginTop: "20px",
    border: "1px solid #e2e8f0",
    borderRadius: "12px",
    overflow: "hidden",
  },
  tableHeader: {
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: "10px",
    padding: "12px 15px",
    backgroundColor: "#334155",
    color: "#fff",
    fontWeight: "600",
    fontSize: "13px",
  },
  tableHeaderCell: {
    textAlign: "left",
  },
  tableRow: {
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: "10px",
    padding: "12px 15px",
    borderBottom: "1px solid #e2e8f0",
    backgroundColor: "#fff",
  },
  tableCellDisease: {
    fontSize: "14px",
    color: "#1e293b",
    fontWeight: "500",
  },
  tableCellCount: {
    fontSize: "14px",
    color: "#475569",
    fontWeight: "600",
  },
  logCard: {
    padding: "5px",
  },
  logHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "15px",
    paddingBottom: "15px",
    borderBottom: "2px solid #e2e8f0",
  },
  logTitle: {
    margin: 0,
    fontSize: "16px",
    color: "#1e293b",
    fontWeight: "600",
  },
  statusBadge: {
    padding: "6px 14px",
    borderRadius: "20px",
    fontSize: "12px",
    fontWeight: "700",
    letterSpacing: "0.5px",
  },
  logDetails: {
    marginBottom: "20px",
  },
  logText: {
    margin: "8px 0",
    fontSize: "14px",
    color: "#64748b",
    fontWeight: "500",
  },
  timestampBox: {
    marginTop: "12px",
    padding: "12px",
    backgroundColor: "#f1f5f9",
    borderRadius: "8px",
    borderLeft: "4px solid #a855f7",
  },
  timestampLabel: {
    fontSize: "12px",
    color: "#64748b",
    fontWeight: "600",
    display: "block",
    marginBottom: "4px",
  },
  timestampValue: {
    fontSize: "13px",
    color: "#1e293b",
    fontWeight: "500",
  },
}

export default ScanVideo

