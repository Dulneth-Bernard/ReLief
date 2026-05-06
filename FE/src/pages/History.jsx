import React, { useState, useEffect } from 'react';
import './History.css';
import client from '../api/client';

const History = () => {
    const [historyData, setHistoryData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            setLoading(true);
            const response = await client.history.getHistory();
            if (response.success) {
                setHistoryData(response.data);
            } else {
                setError('Failed to fetch history');
            }
        } catch (err) {
            setError(err.message || 'An error occurred while fetching history');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="history-container loading-state">
                <div className="loader"></div>
                <p>Loading your analysis history...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="history-container error-state">
                <p className="error-message">✕ {error}</p>
                <button className="retry-btn" onClick={fetchHistory}>Retry</button>
            </div>
        );
    }

    return (
        <div className="history-container">
            <header className="history-header">
                <div className="header-content">
                    <h2>Analysis History</h2>
                    <p>Review your past predictions and explainability maps.</p>
                </div>
            </header>

            {historyData.length === 0 ? (
                <div className="history-empty-state">
                    <p>No history found. Upload an image and run a prediction to see it here.</p>
                </div>
            ) : (
                <div className="history-grid">
                    {historyData.map((item) => (
                        <div key={item.id} className="history-card">
                            <div className="history-card-images">
                                <div className="history-image-wrapper">
                                    <span className="history-image-label">Original</span>
                                    {item.image_url ? (
                                        <img src={item.image_url} alt="Original Lesion" className="history-image" />
                                    ) : (
                                        <div className="history-placeholder-img">No Image Available</div>
                                    )}
                                </div>
                                <div className="history-image-wrapper">
                                    <span className="history-image-label">Grad-CAM</span>
                                    {item.gradcam_url ? (
                                        <img src={item.gradcam_url} alt="Grad-CAM" className="history-image" />
                                    ) : (
                                        <div className="history-placeholder-img">Not Generated</div>
                                    )}
                                </div>
                            </div>
                            <div className="history-card-details">
                                {/* Row 1: model + prediction as plain text */}
                                <div className="history-chips-row">
                                    <span className="history-detail-label">Model:</span>
                                    <span className="history-plain-value">
                                        {item.model_id || 'Unknown'}
                                    </span>
                                </div>
                                <div className="history-chips-row">
                                    <span className="history-detail-label">Prediction:</span>
                                    <span className={`history-plain-value ${item.prediction ? 'prediction-positive' : 'prediction-pending'}`}>
                                        {item.prediction || 'Pending'}
                                    </span>
                                </div>
                                {/* Row 2: confidence + date */}
                                <div className="history-chips-row history-chips-footer">
                                    {item.confidence && (
                                        <span className="confidence-value">
                                            {(item.confidence * 100).toFixed(2)}% conf.
                                        </span>
                                    )}
                                    <span className="date-value" style={{ marginLeft: 'auto' }}>
                                        {new Date(item.created_at).toLocaleString()}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default History;
