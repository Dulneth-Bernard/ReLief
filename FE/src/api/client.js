/**
 * API Client for ReLief Backend
 */

const API_BASE = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;

    const config = {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, config);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'API request failed');
        }

        return data;
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

/**
 * Classification Models API
 */
export const classificationApi = {
    getModels: () => apiFetch('/classification-models'),
    getModel: (id) => apiFetch(`/classification-models/${id}`),
};

/**
 * Diffusion Models API
 */
export const diffusionApi = {
    getModels: () => apiFetch('/diffusion-models'),
    getModel: (id) => apiFetch(`/diffusion-models/${id}`),
};

/**
 * Images API
 */
export const imagesApi = {
    getImages: () => apiFetch('/images'),
    getSampleImages: () => apiFetch('/sample-images'),

    upload: async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/images/upload`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Upload failed');
        }
        return data;
    },
};

/**
 * Prediction API
 */
export const predictionApi = {
    predict: (imageId, modelId) =>
        apiFetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ image_id: imageId, model_id: modelId }),
        }),
};

/**
 * Explainability API
 */
export const explainabilityApi = {
    explain: (imageId, modelId, targetClass = null, method = 'gradcam') =>
        apiFetch('/explain', {
            method: 'POST',
            body: JSON.stringify({
                image_id: imageId,
                model_id: modelId,
                target_class: targetClass,
                method: method
            }),
        }),
};

/**
 * Utility API
 */
export const utilityApi = {
    getClasses: () => apiFetch('/classes'),
    healthCheck: () => apiFetch('/health'),
};
/**
 * Synthetic Images API
 */
export const syntheticApi = {
    getImages: (folder = 'synthetic', nextCursor = null, maxResults = 24) => {
        let url = `/synthetic-images?folder=${encodeURIComponent(folder)}&max_results=${maxResults}`;
        if (nextCursor) {
            url += `&next_cursor=${encodeURIComponent(nextCursor)}`;
        }
        return apiFetch(url);
    }
};

/**
 * History API
 */
export const historyApi = {
    getHistory: () => apiFetch('/history'),
};

export default {
    classification: classificationApi,
    diffusion: diffusionApi,
    images: imagesApi,
    prediction: predictionApi,
    explainability: explainabilityApi,
    utility: utilityApi,
    synthetic: syntheticApi,
    history: historyApi
};
