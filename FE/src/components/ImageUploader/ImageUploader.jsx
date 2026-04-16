import { useRef, useState, useEffect } from 'react'
import { imagesApi } from '../../api/client'
import './ImageUploader.css'

function ImageUploader({ onUpload, uploading = false }) {
    const fileInputRef = useRef(null)
    const [dragOver, setDragOver] = useState(false)
    const [preview, setPreview] = useState(null)
    const [activeTab, setActiveTab] = useState('upload') // 'upload' or 'samples'
    const [error, setError] = useState(null) // Local error for invalid formats
    
    const [sampleImages, setSampleImages] = useState([])
    const [loadingSamples, setLoadingSamples] = useState(false)

    useEffect(() => {
        if (activeTab === 'samples' && sampleImages.length === 0) {
            const loadSamples = async () => {
                setLoadingSamples(true)
                try {
                    const res = await imagesApi.getSampleImages()
                    setSampleImages(res.data || [])
                } catch(e) {
                    console.error("Failed to load sample images", e)
                } finally {
                    setLoadingSamples(false)
                }
            }
            loadSamples()
        }
    }, [activeTab, sampleImages.length])

    const handleFileSelect = (file) => {
        setError(null)
        if (file) {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader()
                reader.onload = (e) => setPreview(e.target.result)
                reader.readAsDataURL(file)
                onUpload(file)
            } else {
                setError("Please upload a valid image file (JPG, PNG). PDFs and other documents are strictly not supported.")
            }
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files[0]
        handleFileSelect(file)
    }

    const handleDragOver = (e) => {
        e.preventDefault()
        setDragOver(true)
    }

    const handleDragLeave = () => {
        setDragOver(false)
    }

    const handleInputChange = (e) => {
        const file = e.target.files[0]
        handleFileSelect(file)
    }

    const triggerFileInput = () => {
        fileInputRef.current?.click()
    }

    const handleSampleSelect = async (sample) => {
        try {
            setPreview(sample.url) // Quick visual feedback
            const response = await fetch(sample.url)
            const blob = await response.blob()
            const file = new File([blob], sample.filename, { type: blob.type })
            onUpload(file)
        } catch (e) {
            console.error("Failed to process sample image", e)
        }
    }

    return (
        <div className="image-uploader card">
            <div className="card-header" style={{ paddingBottom: '0.5rem' }}>
                <h3 className="card-title" style={{ marginBottom: '1rem' }}>Image Input</h3>
                <div className="uploader-tabs" style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <button 
                        className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                        onClick={() => setActiveTab('upload')}
                        style={{ background: 'none', border: 'none', padding: '0.5rem 0', fontWeight: '500', color: activeTab === 'upload' ? 'var(--primary-color)' : 'var(--text-muted)', borderBottom: activeTab === 'upload' ? '2px solid var(--primary-color)' : '2px solid transparent', cursor: 'pointer' }}
                    >
                        Upload Local
                    </button>
                    <button 
                        className={`tab-btn ${activeTab === 'samples' ? 'active' : ''}`}
                        onClick={() => setActiveTab('samples')}
                        style={{ background: 'none', border: 'none', padding: '0.5rem 0', fontWeight: '500', color: activeTab === 'samples' ? 'var(--primary-color)' : 'var(--text-muted)', borderBottom: activeTab === 'samples' ? '2px solid var(--primary-color)' : '2px solid transparent', cursor: 'pointer' }}
                    >
                        Sample Images
                    </button>
                </div>
            </div>

            <div style={{ padding: '1rem' }}>
                {error && (
                    <div className="error-alert" style={{ marginBottom: '1rem' }}>
                        <strong>Unsupported Error:</strong> {error}
                    </div>
                )}
                {activeTab === 'upload' ? (
                    <div
                        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onClick={triggerFileInput}
                    >
                        <label htmlFor="image-upload-input" className="sr-only">
                            Upload a dermoscopic image file
                        </label>
                        <input
                            id="image-upload-input"
                            ref={fileInputRef}
                            type="file"
                            accept=".jpg,.jpeg,.png,image/*"
                            onChange={handleInputChange}
                            className="sr-only"
                        />

                        {preview ? (
                            <div className="preview-container">
                                <img src={preview} alt="Preview" className="preview-image" />
                                <div className="preview-overlay">
                                    <span>Click to change</span>
                                </div>
                            </div>
                        ) : (
                            <div className="upload-placeholder">
                                <div className="upload-icon">📤</div>
                                <p className="upload-text">
                                    Drag & drop a dermoscopic image or click to browse
                                </p>
                                <p className="upload-hint">
                                    Supported formats: JPG, JPEG, PNG
                                </p>
                            </div>
                        )}

                        {uploading && (
                            <div className="upload-loading">
                                <div className="spinner"></div>
                                <span>Processing...</span>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="samples-container">
                        {loadingSamples ? (
                            <div className="text-center" style={{ padding: '2rem 0' }}>
                                <div className="spinner" style={{ margin: '0 auto', marginBottom: '1rem' }}></div>
                                <p>Loading sample images...</p>
                            </div>
                        ) : sampleImages.length === 0 ? (
                            <div className="text-center text-muted" style={{ padding: '2rem 0' }}>
                                <p>No sample images found.</p>
                                <p className="text-sm">Add images to BE/static/sample-images/ to see them here.</p>
                            </div>
                        ) : (
                            <div className="samples-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: '1rem' }}>
                                {sampleImages.map(sample => (
                                    <div 
                                        key={sample.id} 
                                        onClick={() => handleSampleSelect(sample)}
                                        className="sample-item"
                                        style={{ cursor: 'pointer', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '2px solid transparent', transition: 'all 0.2s', position: 'relative' }}
                                    >
                                        <img src={sample.url} alt={sample.filename} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', display: 'block' }} loading="lazy" />
                                        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.6)', color: 'white', fontSize: '0.7rem', padding: '0.2rem', textAlign: 'center' }}>
                                            Select
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                        
                        {uploading && activeTab === 'samples' && (
                            <div className="upload-loading" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(255,255,255,0.8)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
                                <div className="spinner"></div>
                                <span>Running pipeline...</span>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}

export default ImageUploader
