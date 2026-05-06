import { useState, useEffect } from 'react';
import { syntheticApi } from '../api/client';
import './Gallery.css';

const CLASS_FOLDERS = [
    { id: 'AKIEC', name: 'Actinic Keratosis', color: '#ff7675' },
    { id: 'BCC', name: 'Basal Cell Carcinoma', color: '#74b9ff' },
    { id: 'BKL', name: 'Benign Keratosis', color: '#55efc4' },
    { id: 'DF', name: 'Dermatofibroma', color: '#a29bfe' },
    { id: 'MEL', name: 'Melanoma', color: '#ff9ff3' },
    { id: 'NV', name: 'Melanocytic Nevus', color: '#feca57' },
    { id: 'VASC', name: 'Vascular Lesion', color: '#ff9f43' }
];

const getOptimizedUrl = (url) => {
    if (url && url.includes('/upload/')) {
        // Intercepts the Cloudinary URL to add aggressive compression and scaling
        return url.replace('/upload/', '/upload/f_auto,q_auto,w_400/');
    }
    return url;
};

function Gallery() {
    const [selectedFolder, setSelectedFolder] = useState(null);
    
    // Images State
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    // Pagination state
    const [nextCursor, setNextCursor] = useState(null);
    const [history, setHistory] = useState([]); 
    const [totalCount, setTotalCount] = useState(0);

    const loadImages = async (folderPath, cursor = null, isBack = false) => {
        try {
            setLoading(true);
            setError(null);
            
            const response = await syntheticApi.getImages(folderPath, cursor, 24);
            
            setImages(response.data || []);
            setTotalCount(response.total_count || 0);
            
            if (isBack) {
                // Return to previous cursor
                setHistory(prev => prev.slice(0, -1));
            } else if (cursor) {
                // Add previous cursor to history
                setHistory(prev => [...prev, nextCursor]);
            } else {
                // New initial load
                setHistory([]);
            }
            
            setNextCursor(response.next_cursor);
            
        } catch (err) {
            setError(err.message || 'Failed to load images');
        } finally {
            setLoading(false);
        }
    };

    // Trigger load when a folder is selected
    useEffect(() => {
        if (selectedFolder) {
            const folderPath = `synthetic/${selectedFolder.id}`;
            loadImages(folderPath);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedFolder]);

    const handleNextPage = () => {
        if (nextCursor && selectedFolder) {
            loadImages(`synthetic/${selectedFolder.id}`, nextCursor);
        }
    };

    const handlePrevPage = () => {
        if (history.length > 0 && selectedFolder) {
            const prevCursor = history[history.length - 1];
            loadImages(`synthetic/${selectedFolder.id}`, prevCursor, true);
        }
    };

    const handleBackToFolders = () => {
        setSelectedFolder(null);
        setImages([]);
        setHistory([]);
        setNextCursor(null);
        setTotalCount(0);
        setError(null);
    };

    // Render Folders View
    if (!selectedFolder) {
        return (
            <div className="gallery-page animate-fadeIn">
                <div className="gallery-header">
                    <div>
                        <h2>Synthetic Image Gallery</h2>
                        <p className="text-muted">Select a lesion class below to view its generated synthetic images.</p>
                    </div>
                </div>

                <div className="folders-grid">
                    {CLASS_FOLDERS.map((folder) => (
                        <div 
                            key={folder.id} 
                            className="folder-card card" 
                            onClick={() => setSelectedFolder(folder)}
                        >
                            <div className="folder-icon" style={{ backgroundColor: `${folder.color}20`, color: folder.color }}>
                                📁
                            </div>
                            <div className="folder-info">
                                <h3>{folder.id}</h3>
                                <p className="text-muted text-sm">{folder.name}</p>
                            </div>
                            <div className="folder-arrow">→</div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Render Images View
    return (
        <div className="gallery-page animate-fadeIn">
            <div className="gallery-header">
                <div>
                    <div className="breadcrumb">
                        <span className="back-link" onClick={handleBackToFolders}>
                            &larr; Galleries
                        </span>
                        <span className="separator">/</span>
                        <span className="current">{selectedFolder.id}</span>
                    </div>
                    <h2>{selectedFolder.name} Images</h2>
                </div>
                <div className="controls">
                    <button className="btn btn-secondary" onClick={handleBackToFolders}>
                        Back to Folders
                    </button>
                </div>
            </div>

            {error && (
                <div className="error-alert">
                    <p>{error}</p>
                </div>
            )}

            {loading && (
                <div className="loading-state">
                    <span className="spinner" style={{ width: 40, height: 40, borderTopColor: 'var(--primary-color)' }}></span>
                    <p>Loading images for <strong>{selectedFolder.id}</strong>...</p>
                </div>
            )}
            
            {!loading && !error && images.length === 0 && (
                <div className="empty-state card">
                    <p>No images found in folder "<strong>{selectedFolder.id}</strong>".</p>
                    <p className="text-muted text-sm" style={{ marginTop: '0.5rem' }}>Ensure images have been uploaded to this specific folder in Cloudinary.</p>
                </div>
            )}

            {!loading && !error && images.length > 0 && (
                <div className="gallery-grid">
                    {images.map(img => (
                        <div key={img.id} className="gallery-item card" tabIndex="0" role="article" aria-label={`Synthetic ${selectedFolder.name} image`}>
                            <div className="image-container">
                                <img 
                                    src={getOptimizedUrl(img.url)} 
                                    alt={`Synthetic dermoscopic scan of ${selectedFolder.name} - ${img.public_id}`} 
                                    loading="lazy" 
                                    width="400"
                                    height="400"
                                />
                            </div>
                            <div className="gallery-item-info">
                                <span className="image-id" title={img.public_id}>
                                    {img.public_id.split('/').pop()}
                                </span>
                                <span className="image-date">
                                    {new Date(img.created_at).toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <div className="pagination-controls">
                <button 
                    className="btn btn-secondary" 
                    onClick={handlePrevPage} 
                    disabled={history.length === 0 || loading}
                >
                    &laquo; Previous
                </button>
                <div className="pagination-info text-muted">
                    {totalCount > 0 && <span>Total: {totalCount} images found</span>}
                </div>
                <button 
                    className="btn btn-primary" 
                    onClick={handleNextPage} 
                    disabled={!nextCursor || loading}
                >
                    Next &raquo;
                </button>
            </div>
        </div>
    );
}

export default Gallery;
