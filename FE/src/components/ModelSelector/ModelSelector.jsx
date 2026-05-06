import { useState } from 'react'
import './ModelSelector.css'

const MODELS_PER_SLIDE = 2

function getTypeBadgeClass(type = '') {
    const t = type.toLowerCase()
    if (t.includes('final'))       return 'badge-success'
    if (t.includes('advanced'))    return 'badge-success'
    if (t.includes('experimental'))return 'badge-warning'
    if (t.includes('two-stage'))   return 'badge-primary'
    return 'badge-secondary'
}

function ModelSelector({
    title,
    models,
    selectedModel,
    onSelect,
    loading = false,
    error = null
}) {
    const [slide, setSlide] = useState(0)

    if (loading) {
        return (
            <div className="model-selector card">
                <div className="model-selector-loading">
                    <div className="spinner"></div>
                    <p>Loading models...</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="model-selector card">
                <div className="model-selector-error">
                    <p className="text-danger">Error: {error}</p>
                </div>
            </div>
        )
    }

    const totalSlides = Math.ceil(models.length / MODELS_PER_SLIDE)
    const slideModels = models.slice(slide * MODELS_PER_SLIDE, slide * MODELS_PER_SLIDE + MODELS_PER_SLIDE)

    const prev = () => setSlide(s => Math.max(0, s - 1))
    const next = () => setSlide(s => Math.min(totalSlides - 1, s + 1))

    return (
        <div className="model-selector card">
            <div className="card-header">
                <h3 className="card-title">{title}</h3>
                <div className="model-selector-header-right">
                    {selectedModel && (
                        <span className="badge badge-success">✓ Selected</span>
                    )}
                    <span className="carousel-counter">
                        {slide + 1} / {totalSlides}
                    </span>
                </div>
            </div>

            <div className="carousel-wrapper">
                {/* Slide */}
                <div className="carousel-slide">
                    {slideModels.map((model) => (
                        <div
                            key={model.id}
                            className={`model-item ${selectedModel?.id === model.id ? 'selected' : ''}`}
                            onClick={() => onSelect(model)}
                            role="button"
                            aria-pressed={selectedModel?.id === model.id}
                            tabIndex={0}
                            onKeyDown={e => e.key === 'Enter' && onSelect(model)}
                        >
                            <div className="model-item-header">
                                <h4 className="model-name">{model.name}</h4>
                                {model.accuracy && (
                                    <span className="badge badge-primary model-acc-badge">
                                        {model.accuracy.split('/')[0].trim()}
                                    </span>
                                )}
                            </div>

                            <p className="model-description">{model.description}</p>

                            <div className="model-meta">
                                <span className="model-arch">
                                    <strong>Architecture:</strong> {model.architecture}
                                </span>
                                {model.type && (
                                    <span className={`badge ${getTypeBadgeClass(model.type)}`}>
                                        {model.type}
                                    </span>
                                )}
                            </div>

                            {model.accuracy && model.accuracy.includes('/') && (
                                <div className="model-metrics-row">
                                    {model.accuracy.split('/').map((m, i) => (
                                        <span key={i} className="metric-chip">{m.trim()}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* Nav Row */}
                <div className="carousel-nav">
                    <button
                        className="carousel-btn"
                        onClick={prev}
                        disabled={slide === 0}
                        aria-label="Previous models"
                    >
                        ‹
                    </button>

                    <div className="carousel-dots">
                        {Array.from({ length: totalSlides }).map((_, i) => (
                            <button
                                key={i}
                                className={`carousel-dot ${i === slide ? 'active' : ''}`}
                                onClick={() => setSlide(i)}
                                aria-label={`Go to slide ${i + 1}`}
                            />
                        ))}
                    </div>

                    <button
                        className="carousel-btn"
                        onClick={next}
                        disabled={slide === totalSlides - 1}
                        aria-label="Next models"
                    >
                        ›
                    </button>
                </div>
            </div>
        </div>
    )
}

export default ModelSelector
