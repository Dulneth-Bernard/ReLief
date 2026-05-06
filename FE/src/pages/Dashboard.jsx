import { useState, useEffect } from 'react'
import ModelSelector from '../components/ModelSelector/ModelSelector'
import ImageUploader from '../components/ImageUploader/ImageUploader'
import PredictionResults from '../components/PredictionResults/PredictionResults'
import { classificationApi, imagesApi, predictionApi, explainabilityApi } from '../api/client'
import './Dashboard.css'

function Dashboard() {
    // State
    const [models, setModels] = useState([])
    const [selectedModel, setSelectedModel] = useState(null)
    const [uploadedImage, setUploadedImage] = useState(null)
    const [prediction, setPrediction] = useState(null)
    const [heatmap, setHeatmap] = useState(null)
    const [explainMethod, setExplainMethod] = useState('gradcam')

    // Loading states
    const [modelsLoading, setModelsLoading] = useState(true)
    const [uploading, setUploading] = useState(false)
    const [analyzing, setAnalyzing] = useState(false)

    // Errors
    const [modelsError, setModelsError] = useState(null)
    const [analysisError, setAnalysisError] = useState(null)

    // Load models on mount
    useEffect(() => {
        loadModels()
    }, [])

    const loadModels = async () => {
        try {
            setModelsLoading(true)
            setModelsError(null)
            const response = await classificationApi.getModels()
            setModels(response.data)
        } catch (error) {
            setModelsError(error.message)
        } finally {
            setModelsLoading(false)
        }
    }

    const handleModelSelect = (model) => {
        setSelectedModel(model)
        // Re-run analysis if we have an image
        if (uploadedImage) {
            runAnalysis(uploadedImage.id, model.id)
        }
    }

    const handleImageUpload = async (file) => {
        try {
            setUploading(true)
            setPrediction(null)
            setHeatmap(null)
            setAnalysisError(null)

            const response = await imagesApi.upload(file)
            setUploadedImage(response.data)

            // Auto-run analysis if model selected
            if (selectedModel) {
                await runAnalysis(response.data.id, selectedModel.id)
            }
        } catch (error) {
            setAnalysisError(error.message)
        } finally {
            setUploading(false)
        }
    }

    const runAnalysis = async (imageId, modelId, method = explainMethod) => {
        try {
            setAnalyzing(true)
            setAnalysisError(null)

            // Run prediction
            const predResponse = await predictionApi.predict(imageId, modelId)
            setPrediction(predResponse.data)

            // Generate heatmap
            const heatmapResponse = await explainabilityApi.explain(imageId, modelId, null, method)
            setHeatmap(heatmapResponse.data)
        } catch (error) {
            setAnalysisError(error.message)
        } finally {
            setAnalyzing(false)
        }
    }

    const handleAnalyze = () => {
        if (uploadedImage && selectedModel) {
            runAnalysis(uploadedImage.id, selectedModel.id, explainMethod)
        }
    }

    const handleExplainMethodChange = (e) => {
        const newMethod = e.target.value
        setExplainMethod(newMethod)
        
        // Auto re-run if we already have results
        if (uploadedImage && selectedModel && heatmap) {
            runAnalysis(uploadedImage.id, selectedModel.id, newMethod)
        }
    }

    const canAnalyze = uploadedImage && selectedModel && !analyzing

    return (
        <div className="dashboard">
            <section className="hero-section">
                <h2 className="hero-title">
                    AI-Powered Dermoscopic Analysis
                </h2>
                <p className="hero-subtitle">
                    Leveraging diffusion-augmented deep learning for accurate skin lesion classification
                </p>
            </section>

            <div className="dashboard-grid">
                {/* Left Column - Controls */}
                <div className="controls-column">
                    <ModelSelector
                        title="Classification Model"
                        models={models}
                        selectedModel={selectedModel}
                        onSelect={handleModelSelect}
                        loading={modelsLoading}
                        error={modelsError}
                    />

                    <ImageUploader
                        onUpload={handleImageUpload}
                        uploading={uploading}
                    />

                    <div className="card model-selector" style={{ marginTop: '1rem', marginBottom: '1rem' }}>
                        <div className="card-header">
                            <h3 className="card-title" style={{ fontSize: '1rem' }}>Explainability Engine</h3>
                        </div>
                        <div className="model-list" style={{ marginTop: '0.5rem', gap: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                            <div 
                                className={`model-item ${explainMethod === 'gradcam' ? 'selected' : ''}`}
                                onClick={() => handleExplainMethodChange({target: {value: 'gradcam'}})}
                                style={{ padding: '0.75rem 1rem', cursor: 'pointer' }}
                            >
                                <div className="model-item-header">
                                    <h4 className="model-name" style={{ fontSize: '0.95rem', margin: 0 }}>Grad-CAM</h4>
                                    <span className="badge badge-secondary" style={{ fontSize: '0.7rem' }}>Standard</span>
                                </div>
                                <p className="text-muted text-sm" style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem' }}>Fast class activation mapping.</p>
                            </div>
                            
                            <div 
                                className={`model-item ${explainMethod === 'gradcam++' ? 'selected' : ''}`}
                                onClick={() => handleExplainMethodChange({target: {value: 'gradcam++'}})}
                                style={{ padding: '0.75rem 1rem', cursor: 'pointer' }}
                            >
                                <div className="model-item-header">
                                    <h4 className="model-name" style={{ fontSize: '0.95rem', margin: 0 }}>Grad-CAM++</h4>
                                    <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>Advanced</span>
                                </div>
                                <p className="text-muted text-sm" style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem' }}>Sharper contours &amp; better localization.</p>
                            </div>
                        </div>
                    </div>

                    {analysisError && (
                        <div className="error-alert">
                            <strong>Error:</strong> {analysisError}
                        </div>
                    )}

                    <button
                        className="btn btn-primary btn-lg analyze-btn"
                        onClick={handleAnalyze}
                        disabled={!canAnalyze}
                    >
                        {analyzing ? (
                            <>
                                <span className="spinner" style={{ width: 20, height: 20 }}></span>
                                Analyzing...
                            </>
                        ) : (
                            <>🔍 Analyze Image</>
                        )}
                    </button>
                </div>

                {/* Right Column - Results */}
                <div className="results-column">
                    <PredictionResults
                        prediction={prediction}
                        heatmap={heatmap}
                        loading={analyzing}
                    />
                </div>
            </div>

            {/* Model Info Section */}
            {selectedModel && (
                <section className="model-info-section animate-fadeIn">
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">Selected Model Summary</h3>
                        </div>
                        <div className="model-info-grid">
                            <div className="info-item">
                                <span className="info-label">Model</span>
                                <span className="info-value">{selectedModel.name}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Architecture</span>
                                <span className="info-value">{selectedModel.architecture}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Accuracy</span>
                                <span className="info-value text-success">
                                    {selectedModel.accuracy || 'N/A'}
                                </span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Type</span>
                                <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                                    <span className={`badge ${
                                        selectedModel.type?.toLowerCase().includes('final') ? 'badge-success' :
                                        selectedModel.type?.toLowerCase().includes('advanced') ? 'badge-success' :
                                        selectedModel.type?.toLowerCase().includes('experimental') ? 'badge-warning' :
                                        selectedModel.type?.toLowerCase().includes('two-stage') ? 'badge-primary' :
                                        'badge-secondary'
                                    }`} style={{ whiteSpace: 'nowrap', alignSelf: 'flex-start' }}>
                                        {selectedModel.type || 'Baseline'}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <p className="model-description-text">{selectedModel.description}</p>
                    </div>
                </section>
            )}
        </div>
    )
}

export default Dashboard
