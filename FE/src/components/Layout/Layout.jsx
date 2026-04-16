import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import './Layout.css'

function Layout({ children }) {
    const [isMenuOpen, setIsMenuOpen] = useState(false)

    return (
        <div className="layout">
            <header className="header">
                <div className="container">
                    <div className="header-content">
                        <div className="logo">
                            <span className="logo-icon">🔬</span>
                            <div className="logo-text">
                                <h1>ReLief</h1>
                                <span className="logo-subtitle">Dermoscopic Analysis Platform</span>
                            </div>
                        </div>
                        
                        <button 
                            className="mobile-menu-toggle" 
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            aria-label="Toggle Navigation"
                        >
                            {isMenuOpen ? '✕' : '☰'}
                        </button>

                        <nav className={`nav ${isMenuOpen ? 'open' : ''}`}>
                            <NavLink to="/" className={({isActive}) => isActive ? "nav-link active" : "nav-link"} onClick={() => setIsMenuOpen(false)} end>Dashboard</NavLink>
                            <NavLink to="/gallery" className={({isActive}) => isActive ? "nav-link active" : "nav-link"} onClick={() => setIsMenuOpen(false)}>Synthetic Gallery</NavLink>
                            <NavLink to="/history" className={({isActive}) => isActive ? "nav-link active" : "nav-link"} onClick={() => setIsMenuOpen(false)}>History</NavLink>
                        </nav>
                    </div>
                </div>
            </header>

            <main className="main-content">
                <div className="container">
                    {children}
                </div>
            </main>

            <footer className="footer">
                <div className="container">
                    <p className="text-muted text-sm">
                        ReLief - Diffusion-Augmented Deep Learning for Dermoscopic Analysis
                    </p>
                </div>
            </footer>
        </div>
    )
}

export default Layout
