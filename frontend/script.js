// FactTrack 2.0 - Impressive UI JavaScript

// DOM Elements
const form = document.getElementById('analysisForm');
const textarea = document.getElementById('articleText');
const fileUpload = document.getElementById('fileUpload');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const resultsSummary = document.getElementById('resultsSummary');
const resultsList = document.getElementById('resultsList');
const errorMessage = document.getElementById('errorMessage');
const statusIndicator = document.getElementById('statusIndicator');
const modelBadge = document.getElementById('modelBadge');
const modelInfo = document.getElementById('modelInfo');
const footerStats = document.getElementById('footerStats');
const copyResultsBtn = document.getElementById('copyResultsBtn');
const downloadResultsBtn = document.getElementById('downloadResultsBtn');

// Global variables
let currentResults = null;
let systemInfo = {};
const API_BASE = window.location.origin;

// Category color palette (HSL for better color distribution)
const categoryColors = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    addTypewriterEffect();
});

/**
 * Initialize application
 */
function initializeApp() {
    checkSystemHealth();
    loadModelInfo();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    form.addEventListener('submit', handleFormSubmit);
    clearBtn.addEventListener('click', clearForm);
    fileUpload.addEventListener('change', handleFileUpload);
    textarea.addEventListener('input', updateCharCount);
    copyResultsBtn.addEventListener('click', copyResults);
    downloadResultsBtn.addEventListener('click', downloadResults);
    
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', smoothScroll);
    });
}

/**
 * Typewriter effect for placeholder (subtle enhancement)
 */
function addTypewriterEffect() {
    const placeholder = textarea.getAttribute('placeholder');
    // Keep the placeholder static for better UX
}

/**
 * Check system health and model status
 */
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        if (data.models_loaded) {
            setSystemStatus('ready', 'System Ready');
            analyzeBtn.disabled = false;
            
            systemInfo = data.model_info || {};
            updateModelBadge(systemInfo);
            
            // Animate status indicator
            setTimeout(() => {
                statusIndicator.classList.add('pulse-success');
            }, 500);
        } else {
            setSystemStatus('error', 'Models Not Loaded');
            analyzeBtn.disabled = true;
            showError('System not ready. Please train the models first:\n\n1. python download_data.py\n2. python train.py\n3. python app.py');
        }
    } catch (error) {
        setSystemStatus('error', 'Connection Error');
        analyzeBtn.disabled = true;
        showError('Unable to connect to the server. Please ensure the Flask app is running on port 5000.');
        console.error('Connection error:', error);
    }
}

/**
 * Set system status
 */
function setSystemStatus(status, message) {
    statusIndicator.classList.remove('ready', 'error');
    statusIndicator.classList.add(status);
    statusIndicator.querySelector('.status-text').textContent = message;
}

/**
 * Update model badge
 */
function updateModelBadge(info) {
    const modelType = info.model_type || 'BERT';
    const numCategories = info.num_categories || 20;
    modelBadge.textContent = `${modelType} • ${numCategories} Categories`;
}

/**
 * Load detailed model information
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/model-info`);
        const data = await response.json();
        
        if (data.success) {
            systemInfo = { ...systemInfo, ...data.model_info };
            
            // Generate category colors
            if (data.model_info && data.model_info.categories) {
                generateCategoryColors(data.model_info.categories);
            }
            
            // Update footer stats
            if (data.metrics) {
                updateFooterStats(data.metrics);
            }
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

/**
 * Generate beautiful colors for categories
 */
function generateCategoryColors(categories) {
    const goldenRatio = 0.618033988749895;
    let hue = Math.random();
    
    categories.forEach(category => {
        hue += goldenRatio;
        hue %= 1;
        const h = Math.floor(hue * 360);
        categoryColors[category] = {
            color: `hsl(${h}, 70%, 60%)`,
            shadow: `hsla(${h}, 70%, 60%, 0.3)`
        };
    });
}

/**
 * Update footer statistics
 */
function updateFooterStats(metrics) {
    const catAcc = metrics.category?.accuracy;
    const biasF1 = metrics.bias?.f1_macro;
    
    if (catAcc && biasF1) {
        footerStats.innerHTML = `
            <strong>Model Performance:</strong> 
            Category <span style="color: var(--success-color);">${(catAcc * 100).toFixed(1)}%</span> • 
            Bias F1 <span style="color: var(--info-color);">${(biasF1 * 100).toFixed(1)}%</span>
        `;
    }
}

/**
 * Handle form submission with smooth transitions
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const text = textarea.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (text.length < 30) {
        showError('Please enter at least 30 characters for accurate analysis.');
        return;
    }
    
    // Smooth transition to loading
    hideError();
    hideResults();
    
    // Animate out input section
    document.querySelector('.input-section').style.opacity = '0.5';
    
    setTimeout(() => {
        showLoading();
    }, 200);
    
    try {
        const startTime = Date.now();
        
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        // Ensure minimum loading time for smooth UX
        const elapsed = Date.now() - startTime;
        const minDelay = 1000;
        
        if (elapsed < minDelay) {
            await new Promise(resolve => setTimeout(resolve, minDelay - elapsed));
        }
        
        if (data.success) {
            currentResults = data;
            displayResults(data);
            
            // Animate in results
            setTimeout(() => {
                document.querySelector('.input-section').style.opacity = '1';
            }, 300);
        } else {
            showError(data.error || 'An error occurred during analysis. Please try again.');
            document.querySelector('.input-section').style.opacity = '1';
        }
    } catch (error) {
        showError('Failed to connect to the server. Please ensure the Flask app is running.');
        console.error('Analysis error:', error);
        document.querySelector('.input-section').style.opacity = '1';
    } finally {
        hideLoading();
    }
}

/**
 * Display results with impressive animations
 */
function displayResults(data) {
    const { results, total_paragraphs, processing_time_ms, model } = data;
    
    // Calculate statistics
    const categoryCount = {};
    let totalBiasProbability = 0;
    let biasedCount = 0;
    let highConfidenceCount = 0;
    
    results.forEach(result => {
        if (result.top_categories && result.top_categories[0]) {
            const primaryCat = result.top_categories[0].category;
            categoryCount[primaryCat] = (categoryCount[primaryCat] || 0) + 1;
        }
        
        totalBiasProbability += result.bias.probability;
        if (result.bias.label === 'biased') biasedCount++;
        if (result.bias.confidence_level === 'high') highConfidenceCount++;
    });
    
    const topCategory = Object.entries(categoryCount).sort((a, b) => b[1] - a[1])[0];
    const avgBias = (totalBiasProbability / total_paragraphs * 100).toFixed(1);
    
    // Create summary cards with animations
    resultsSummary.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Paragraphs</div>
            <div class="stat-value">${total_paragraphs}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Primary Topic</div>
            <div class="stat-value" style="font-size: 1.5em; text-transform: capitalize;">
                ${topCategory ? topCategory[0].replace('_', ' ') : 'Mixed'}
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Bias Risk</div>
            <div class="stat-value" style="color: ${getBiasColorGradient(avgBias / 100)};">
                ${avgBias}%
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Processing Time</div>
            <div class="stat-value" style="font-size: 1.3em;">
                ${processing_time_ms}<span style="font-size: 0.5em;">ms</span>
            </div>
        </div>
    `;
    
    // Create result cards
    resultsList.innerHTML = results.map((result, index) => {
        const biasClass = getBiasClass(result.bias.probability);
        const biasPercent = (result.bias.probability * 100).toFixed(1);
        const biasColor = getBiasColorGradient(result.bias.probability);
        
        // Top-3 category badges
        const categoryBadges = result.top_categories.slice(0, 3).map((cat, idx) => {
            const colors = categoryColors[cat.category] || { color: '#667eea', shadow: 'rgba(102, 126, 234, 0.3)' };
            const confidence = (cat.confidence * 100).toFixed(1);
            const rankEmoji = idx === 0 ? '🥇' : idx === 1 ? '🥈' : '🥉';
            
            return `
                <div class="category-badge category-rank-${idx + 1}" 
                     style="border-color: ${colors.color}; color: ${colors.color}; box-shadow: 0 4px 12px ${colors.shadow};"
                     title="${cat.category}: ${confidence}% confidence">
                    <span>${rankEmoji}</span>
                    <span>${cat.category.replace('_', ' ')}</span>
                    <span style="opacity: 0.7; font-weight: 500;">${confidence}%</span>
                </div>
            `;
        }).join('');
        
        // Bias indicators
        const indicators = result.bias.indicators && result.bias.indicators.length > 0
            ? `<div class="bias-indicators">
                <strong>⚠️ Bias Indicators:</strong> ${result.bias.indicators.map(ind => `<code>${ind}</code>`).join(', ')}
               </div>`
            : '';
        
        return `
            <div class="result-card ${biasClass}">
                <div class="category-badges">
                    ${categoryBadges}
                </div>
                
                <div class="bias-indicator">
                    <div class="bias-header">
                        <span class="bias-label ${result.bias.label}">
                            ${result.bias.label === 'biased' ? '⚠️' : '✅'} 
                            ${result.bias.label.replace('_', ' ')}
                        </span>
                        <span class="bias-confidence ${result.bias.confidence_level}">
                            ${result.bias.confidence_level} confidence
                        </span>
                    </div>
                    <div class="bias-probability">
                        <div class="bias-bar-container">
                            <div class="bias-bar" 
                                 style="width: ${biasPercent}%; background: ${biasColor};">
                            </div>
                        </div>
                        <span style="font-weight: 700; min-width: 50px; text-align: right; color: ${biasColor};">
                            ${biasPercent}%
                        </span>
                    </div>
                    ${indicators}
                </div>
                
                <div class="result-text">${escapeHtml(result.paragraph)}</div>
                
                <div class="result-meta">
                    <span>📄 Paragraph ${index + 1}/${total_paragraphs}</span>
                    <span>🎯 ${result.top_categories[0].category.replace('_', ' ')}</span>
                    <span>⚖️ Bias: ${biasPercent}%</span>
                    <span>🔒 ${result.bias.confidence_level} confidence</span>
                </div>
            </div>
        `;
    }).join('');
    
    showResults();
}

/**
 * Get bias class for card styling
 */
function getBiasClass(probability) {
    if (probability > 0.6) return 'bias-high';
    if (probability > 0.3) return 'bias-medium';
    return 'bias-low';
}

/**
 * Get color gradient for bias visualization
 */
function getBiasColorGradient(probability) {
    // Green (low) → Yellow (medium) → Red (high)
    if (probability <= 0.5) {
        // Green to Yellow
        const r = Math.floor(16 + (245 - 16) * (probability * 2));
        const g = Math.floor(185 + (158 - 185) * (probability * 2));
        const b = Math.floor(129 + (11 - 129) * (probability * 2));
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // Yellow to Red
        const r = Math.floor(245 + (239 - 245) * ((probability - 0.5) * 2));
        const g = Math.floor(158 - 158 * ((probability - 0.5) * 2));
        const b = Math.floor(11 + (68 - 11) * ((probability - 0.5) * 2));
        return `rgb(${r}, ${g}, ${b})`;
    }
}

/**
 * Handle file upload with visual feedback
 */
function handleFileUpload(e) {
    const file = e.target.files[0];
    
    if (!file) return;
    
    if (file.type !== 'text/plain') {
        showError('Please upload a .txt file');
        e.target.value = '';
        return;
    }
    
    // Show loading feedback
    const uploadBtn = document.querySelector('.file-upload-btn');
    uploadBtn.style.background = 'rgba(102, 126, 234, 0.3)';
    
    const reader = new FileReader();
    
    reader.onload = (event) => {
        textarea.value = event.target.result;
        updateCharCount();
        hideError();
        
        // Success feedback
        uploadBtn.style.background = 'rgba(16, 185, 129, 0.2)';
        setTimeout(() => {
            uploadBtn.style.background = '';
        }, 2000);
        
        // Smooth focus
        textarea.focus();
    };
    
    reader.onerror = () => {
        showError('Failed to read file');
        uploadBtn.style.background = '';
    };
    
    reader.readAsText(file);
}

/**
 * Update character count with color transitions
 */
function updateCharCount() {
    const count = textarea.value.length;
    charCount.textContent = count.toLocaleString();
    
    // Color transitions based on length
    const counter = document.querySelector('.char-counter');
    if (count > 50000) {
        counter.style.background = 'rgba(239, 68, 68, 0.2)';
        counter.style.borderColor = 'rgba(239, 68, 68, 0.5)';
    } else if (count > 1000) {
        counter.style.background = 'rgba(16, 185, 129, 0.2)';
        counter.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else {
        counter.style.background = 'rgba(102, 126, 234, 0.2)';
        counter.style.borderColor = 'rgba(102, 126, 234, 0.3)';
    }
}

/**
 * Clear form with animation
 */
function clearForm() {
    // Fade out
    textarea.style.opacity = '0';
    
    setTimeout(() => {
        textarea.value = '';
        fileUpload.value = '';
        updateCharCount();
        hideResults();
        hideError();
        textarea.style.opacity = '1';
    }, 200);
}

/**
 * Copy results with visual feedback
 */
async function copyResults() {
    if (!currentResults) return;
    
    try {
        const text = currentResults.results.map((result, i) => {
            const topCats = result.top_categories.map((c, idx) => 
                `${idx + 1}. ${c.category} (${(c.confidence * 100).toFixed(1)}%)`
            ).join('\n     ');
            
            const indicators = result.bias.indicators && result.bias.indicators.length > 0 
                ? `\n   Indicators: ${result.bias.indicators.join(', ')}` 
                : '';
            
            return `
${'='.repeat(80)}
PARAGRAPH ${i + 1}
${'='.repeat(80)}

${result.paragraph}

TOP CATEGORIES:
     ${topCats}

BIAS ANALYSIS:
   Label: ${result.bias.label}
   Probability: ${(result.bias.probability * 100).toFixed(1)}%
   Confidence: ${result.bias.confidence_level}${indicators}

`;
        }).join('\n');
        
        await navigator.clipboard.writeText(text);
        
        // Animated success feedback
        const btn = copyResultsBtn;
        btn.style.background = 'rgba(16, 185, 129, 0.2)';
        btn.style.color = 'var(--success-color)';
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
            </svg>
        `;
        
        setTimeout(() => {
            btn.style.background = '';
            btn.style.color = '';
            btn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"/>
                    <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z"/>
                </svg>
            `;
        }, 2000);
    } catch (error) {
        showError('Failed to copy to clipboard. Please try again.');
    }
}

/**
 * Download results as JSON with animation
 */
function downloadResults() {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `facttrack-analysis-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    // Success feedback
    const btn = downloadResultsBtn;
    btn.style.background = 'rgba(16, 185, 129, 0.2)';
    btn.style.color = 'var(--success-color)';
    
    setTimeout(() => {
        btn.style.background = '';
        btn.style.color = '';
    }, 2000);
}

/**
 * Show/hide UI elements with smooth transitions
 */
function showLoading() {
    loadingState.style.display = 'block';
    setTimeout(() => {
        loadingState.style.opacity = '1';
    }, 10);
}

function hideLoading() {
    loadingState.style.opacity = '0';
    setTimeout(() => {
        loadingState.style.display = 'none';
    }, 300);
}

function showResults() {
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.style.opacity = '1';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 10);
}

function hideResults() {
    resultsSection.style.opacity = '0';
    setTimeout(() => {
        resultsSection.style.display = 'none';
        currentResults = null;
    }, 300);
}

function showError(message) {
    const errorText = errorMessage.querySelector('.error-text');
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.opacity = '1';
    }, 10);
    
    // Scroll to error
    setTimeout(() => {
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

function hideError() {
    errorMessage.style.opacity = '0';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 300);
}

/**
 * Smooth scroll
 */
function smoothScroll(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Add smooth transitions to opacity changes
 */
const style = document.createElement('style');
style.textContent = `
    .input-section,
    #loadingState,
    #resultsSection,
    #errorMessage {
        opacity: 1;
        transition: opacity 0.3s ease;
    }
`;
document.head.appendChild(style);

// Add loading class for initial state
document.body.classList.add('loading');
setTimeout(() => {
    document.body.classList.remove('loading');
}, 100);

console.log('%c FactTrack 2.0 ', 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 20px; padding: 10px 20px; border-radius: 5px; font-weight: bold;');
console.log('%c BERT-Powered News Analysis System ', 'color: #667eea; font-size: 14px; font-weight: bold;');
console.log('%c Ready for AI-powered analysis! ', 'color: #10b981; font-size: 12px;');
