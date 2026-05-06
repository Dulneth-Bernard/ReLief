import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'

import Layout from './components/Layout/Layout'
import Dashboard from './pages/Dashboard'
import Gallery from './pages/Gallery'
import History from './pages/History'

function App() {
    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/gallery" element={<Gallery />} />
                    <Route path="/history" element={<History />} />
                </Routes>
            </Layout>
        </Router>
    )
}

export default App
