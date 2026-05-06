import { useRef, useState } from 'react'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import './PredictionResults.css'

const CLASS_COLORS = {
    'NV': '#22c55e',    // Green - benign
    'MEL': '#ef4444',   // Red - malignant
    'BKL': '#3b82f6',   // Blue
    'BCC': '#f97316',   // Orange
    'AKIEC': '#eab308', // Yellow
    'VASC': '#a855f7',  // Purple
    'DF': '#06b6d4',    // Cyan
}

const CLASS_NAMES = {
    'NV': 'Melanocytic Nevus',
    'MEL': 'Melanoma',
    'BKL': 'Benign Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratosis',
    'VASC': 'Vascular Lesion',
    'DF': 'Dermatofibroma',
}

function PredictionResults({ prediction, heatmap, loading = false }) {
    const reportRef = useRef(null)
    const [isGeneratingPdf, setIsGeneratingPdf] = useState(false)

    const handleDownloadPdf = async () => {
        if (!reportRef.current) return;
        setIsGeneratingPdf(true);
        
        try {
            const canvas = await html2canvas(reportRef.current, {
                scale: 2, 
                useCORS: true,
                backgroundColor: '#ffffff'
            });
            
            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
            
            // Add a small margin (e.g., 10mm)
            const margin = 10;
            const contentWidth = pdfWidth - 2 * margin;
            const contentHeight = (canvas.height * contentWidth) / canvas.width;
            
            pdf.addImage(imgData, 'PNG', margin, margin, contentWidth, contentHeight);
            pdf.save(`ReLief_Report_${prediction.predicted_class}.pdf`);
        } catch (error) {
            console.error("PDF generation failed:", error);
            alert("Failed to generate PDF report.");
        } finally {
            setIsGeneratingPdf(false);
        }
    }
    if (loading) {
        return (
            <div className="prediction-results card">
                <div className="results-loading">
                    <div className="spinner"></div>
                    <p>Analyzing image...</p>
                </div>
            </div>
        )
    }

    if (!prediction) {
        return (
            <div className="prediction-results card">
                <div className="card-header">
                    <h3 className="card-title">Analysis Results</h3>
                </div>
                <div className="results-empty">
                    <p className="text-muted">
                        Upload an image and select a model to see results
                    </p>
                </div>
            </div>
        )
    }

    const isMalignant = ['MEL', 'BCC', 'AKIEC'].includes(prediction.predicted_class)
    const primaryColor = CLASS_COLORS[prediction.predicted_class] || '#3b82f6'

    return (
        <div className="prediction-results card" ref={reportRef} style={{ position: 'relative' }}>
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <h3 className="card-title" style={{ margin: 0 }}>Clinical Analysis Report</h3>
                    <span className={`badge ${isMalignant ? 'badge-danger' : 'badge-success'}`}>
                        {isMalignant ? 'Review Required' : 'Likely Benign'}
                    </span>
                </div>
                
                <button 
                    className="btn btn-secondary" 
                    onClick={handleDownloadPdf}
                    disabled={isGeneratingPdf}
                    data-html2canvas-ignore="true"
                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                >
                    {isGeneratingPdf ? 'Generating...' : '💾 Download PDF'}
                </button>
            </div>

            <div className="primary-prediction">
                <div
                    className="prediction-badge"
                    style={{ background: primaryColor }}
                >
                    <span className="prediction-class">{prediction.predicted_class}</span>
                    <span className="prediction-confidence">
                        {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="prediction-details">
                    <h4>{CLASS_NAMES[prediction.predicted_class] || prediction.predicted_class}</h4>
                    <p className="text-muted text-sm">Primary Diagnosis</p>
                </div>
            </div>

            <div className="probability-chart">
                <h5 className="chart-title">Class Probabilities</h5>
                {Object.entries(prediction.class_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, prob]) => (
                        <div key={cls} className="probability-row">
                            <div className="probability-label">
                                <span
                                    className="probability-indicator"
                                    style={{ background: CLASS_COLORS[cls] }}
                                ></span>
                                <span className="probability-class">{cls}</span>
                                <span className="probability-name text-muted">
                                    {CLASS_NAMES[cls]}
                                </span>
                            </div>
                            <div className="probability-bar-container">
                                <div
                                    className="probability-bar"
                                    style={{
                                        width: `${prob * 100}%`,
                                        background: CLASS_COLORS[cls]
                                    }}
                                ></div>
                                <span className="probability-value">
                                    {(prob * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    ))
                }
            </div>

            {heatmap && (
                <div className="heatmap-section">
                    <h5 className="chart-title">
                        Explainability Heatmap ({heatmap.method === 'gradcam++' ? 'Grad-CAM++' : 'Grad-CAM'})
                    </h5>
                    <div className="heatmap-container">
                        <img
                            src={heatmap.overlay_path}
                            alt="Grad-CAM Heatmap"
                            className="heatmap-image"
                        />
                    </div>
                    <p className="text-muted text-sm text-center mt-sm">
                        Highlighted regions indicate areas influencing the model's decision
                    </p>
                </div>
            )}
        </div>
    )
}

export default PredictionResults
